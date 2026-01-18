import inspect
import logging
import torch
from torch._ops import HigherOrderOperator
from torch.utils.checkpoint import checkpoint, uid
import torch._dynamo.config
class TagActivationCheckpoint(HigherOrderOperator):
    """
    This operator is supposed to be used only with torch.compile stack. This
    accepts a Fx graph module which needs to be checkpointed. This operator adds
    "recomputable" tag to the nodes of the Fx graph that should be recomputed.

    The goal is to:
    1. Avoid using Dynamo to trace through saved tensor hooks.
    2. For selective checkpointing case, let AOTAutograd trace through
       saved tensor hooks but has special logic with TorchDispatchMode to override
       the usual saved_tensor_hooks fn logic in order to tag the nodes.
    3. Rely on the partitioners to actually duplicate the nodes.
    This sits well in the torch.compile stack, because by the time graph
    reaches partitioner, inductor has already run its functionalization of rng
    ops. Therefore, the duplication of nodes, by design, respects the rng states
    in the forward and recomputed forward in backward.
    """

    def __init__(self):
        super().__init__('tag_activation_checkpoint')

    @staticmethod
    def divide_kwargs(kwargs):
        """
        checkpoint fn can have mixed kwargs between checkpointed fn and
        checkpoint fn itself. For example
        >> def gn(x, y, z=None):
        >>     a = torch.matmul(x, y)
        >>     if z is not None:
        >>         return torch.matmul(a, z)
        >>     return a
        >> def fn(x, y, z):
        >>     return torch.cos(checkpoint(gn, x, y, use_reentrant=False, z=z))
        In the above case, z belongs to checkpointed function gn, but
        use_reentrant belongs to the checkpoint function. This function splits
        the kwargs into checkpoint_kwargs and gmod_kwargs (or
        checkpointed_fn_kwargs).
        We do sorting to ensure same graph from run to run for better
        debuggability. It is not required for correctness.
        """
        ckpt_signature = inspect.signature(checkpoint)
        checkpoint_keys = set()
        for name in ckpt_signature.parameters:
            if name in ('function', 'args', 'kwargs'):
                continue
            checkpoint_keys.add(name)
        checkpoint_keys.add('preserve_rng_state')
        checkpoint_kwargs = {name: kwargs[name] for name in kwargs.keys() if name in checkpoint_keys}
        gmod_kwargs = {name: kwargs[name] for name in kwargs.keys() if name not in checkpoint_keys}
        return (checkpoint_kwargs, gmod_kwargs)

    def tag_nodes(self, gmod):
        unique_graph_id = next(uid)
        for node in gmod.graph.nodes:
            if node.op in ('call_function', 'call_method', 'call_module'):
                node.meta['recompute'] = unique_graph_id
        return gmod

    def __call__(self, gmod, *args, **kwargs):
        import torch.fx.traceback as fx_traceback
        from torch.fx import Interpreter
        if '_checkpoint_context_fn' in gmod.meta:
            assert torch._dynamo.config._experimental_support_context_fn_in_torch_utils_checkpoint, 'Passing context_fn to torch.utils.checkpoint is currently not supported under torch.compile'
            log.warning('\nDetected that context_fn is passed to torch.utils.checkpoint under torch.compile.\nPlease make sure the checkpointed region does not contain in-place ops (e.g. torch.relu_).\n')
            kwargs['use_reentrant'] = False
            kwargs['context_fn'] = gmod.meta['_checkpoint_context_fn']
            gmod = self.tag_nodes(gmod)
            with fx_traceback.preserve_node_meta():
                return checkpoint(Interpreter(gmod).run, *args, **kwargs)
        else:
            gmod = self.tag_nodes(gmod)
            with fx_traceback.preserve_node_meta():
                return Interpreter(gmod).run(*args)