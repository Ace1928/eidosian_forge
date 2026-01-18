import collections
import itertools
import logging
import operator
import tempfile
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import (
import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed._spmd.graph_utils import (
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def _split_fused_adam(gm: IterGraphModule, orig_optim_block: FusedOptimizerBlock, split_gradients: Set[fx.Node]) -> Tuple[FusedOptimizerBlock, FusedOptimizerBlock]:
    """Split the `orig_optim_block` into two FusedOptimizerBlock.

    The first one will be the optimizer that optimize `split_gradients`. The second one is
    used to optimize the remaining gradients.
    An assert will be raised if one of the optimizer optimize zero gradients.
    """
    orig_optim_args = AdamArgs(*orig_optim_block.optim.optim_node.args)
    optim_args = (AdamArgs([], [], [], [], [], []), AdamArgs([], [], [], [], [], []))
    orig_optim_indices: Tuple[List[int], List[int]] = ([], [])
    orig_step_indices: Tuple[List[int], List[int]] = ([], [])
    for idx, gradient in enumerate(orig_optim_args.grads):
        group_idx = 0 if gradient in split_gradients else 1
        orig_optim_indices[group_idx].append(idx)
        for orig_arg, optim_arg in zip(orig_optim_args, optim_args[group_idx]):
            if orig_arg:
                optim_arg.append(orig_arg[idx])
        orig_step_output = optim_args[group_idx].state_steps[-1]
        assert str(orig_step_output.target).startswith('aten.copy_'), f'The copy output is {orig_step_output.target}, expect aten.copy_'
        orig_step_getitem = orig_step_output.args[1]
        assert 'getitem' in str(orig_step_getitem.target), f'The copy getitem is {orig_step_getitem.target}, expect operator.getitem'
        orig_step_idx = orig_step_getitem.args[1]
        orig_step_indices[group_idx].append(orig_step_idx)
    if not all((l for l in orig_step_indices + orig_optim_indices)):
        raise ValueError('At least one split optimizer does not have input.')
    output = get_output(gm.graph)
    results: List[FusedOptimizerBlock] = []
    flatten_output_args, spec = tree_flatten((output.args, output.kwargs))
    flatten_output_args_indices: DefaultDict[fx.Node, Set[int]] = collections.defaultdict(set)
    for idx, output_arg in enumerate(flatten_output_args):
        if isinstance(output_arg, fx.Node):
            flatten_output_args_indices[output_arg].add(idx)

    def replace_flatten_output_args(orig_node: fx.Node, new_node: fx.Node):
        for idx in flatten_output_args_indices[orig_node]:
            flatten_output_args[idx] = new_node
    for group_idx in range(2):
        step_args: List[fx.Node] = []
        orig_step_outputs: List[fx.Node] = []
        with gm.graph.inserting_after(orig_optim_block.optim.optim_node):
            for idx in orig_step_indices[group_idx]:
                step_args.append(cast(Tuple[fx.Node, ...], orig_optim_block.step.add_node.args[0])[idx])
                orig_step_outputs.append(orig_optim_block.step.outputs[idx])
            step = gm.graph.call_function(aten._foreach_add.Scalar, (step_args, 1))
        step_block = ForeachAddBlock(step, generate_output=True)
        for i, step_output in enumerate(step_block.outputs):
            orig_step_output = orig_step_outputs[i]
            replace_flatten_output_args(orig_step_output, step_output)
            assert optim_args[group_idx].state_steps[i] == orig_step_output, f'The expected step output node mismatched, {orig_step_output} {optim_args[group_idx].state_steps[i]}'
            optim_args[group_idx].state_steps[i] = step_output
        with gm.graph.inserting_after(step_block.outputs[0]):
            optim = gm.graph.call_function(aten._fused_adam.default, optim_args[group_idx], orig_optim_block.optim.optim_node.kwargs)
        optim_block = FusedAdamBlock(optim, generate_output=True)
        for curr_idx, orig_idx in enumerate(orig_optim_indices[group_idx]):
            list_names = ('param_outputs', 'exp_avgs_outputs', 'exp_avg_sqs_outputs')
            for name in list_names:
                orig_list = getattr(orig_optim_block.optim, name)
                curr_list = getattr(optim_block, name)
                replace_flatten_output_args(orig_list[orig_idx], curr_list[curr_idx])
        results.append(FusedOptimizerBlock(step_block, optim_block))
    output_args, output_kwargs = tree_unflatten(flatten_output_args, spec)
    gm.graph.node_set_args(output, output_args)
    gm.graph.node_set_kwargs(output, output_kwargs)
    for copy_output in itertools.chain(orig_optim_block.optim.param_outputs, orig_optim_block.optim.exp_avgs_outputs, orig_optim_block.optim.exp_avg_sqs_outputs):
        gm.graph.erase_node(copy_output)
    gm.graph.eliminate_dead_code()
    for copy_output in orig_optim_block.step.outputs:
        gm.graph.erase_node(copy_output)
    gm.graph.eliminate_dead_code()
    return (results[0], results[1])