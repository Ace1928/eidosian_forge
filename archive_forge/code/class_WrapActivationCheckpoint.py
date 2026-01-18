import inspect
import logging
import torch
from torch._ops import HigherOrderOperator
from torch.utils.checkpoint import checkpoint, uid
import torch._dynamo.config
class WrapActivationCheckpoint(HigherOrderOperator):
    """
    This operator is used to wrap torch.utils.checkpoint. This avoids
    TorchDynamo to look into saved tensor hooks and directly passes the control
    to AOT Autograd, which is ok with tracing saved tensor hooks. As a result of
    AOT tracing torch.utils.checkpoint code, we have a backward graph with
    recomputed forward nodes.

    However, we might deprecate this operator soon. The difficulty arises in the
    functionalization of rng ops. Today, there are two different
    functionalization of rng ops - one at AOT autograd and other at Inductor.
    And they are difficult to map to each other. The rng states also complicate
    pattern matching in Inductor. Due to the ease of implementation, we are
    currently inclined towards functionalization at Inductor level, which means
    that duplication/recomputation is done as a compiler pass in the
    partitioners. See TagActivationCheckpoint for more information.
    """

    def __init__(self):
        super().__init__('wrap_activation_checkpoint')

    def __call__(self, function, *args, **kwargs):
        import torch.fx.traceback as fx_traceback
        from torch.fx import Interpreter
        kwargs['use_reentrant'] = False
        kwargs['preserve_rng_state'] = False
        with fx_traceback.preserve_node_meta():
            return checkpoint(Interpreter(function).run, *args, **kwargs)