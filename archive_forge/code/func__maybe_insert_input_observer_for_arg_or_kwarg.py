import torch
from torch._subclasses import FakeTensor
from torch.ao.quantization.fx.prepare import (
from torch.fx import (
from torch.fx.node import Argument
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
from typing import Dict, Tuple, Union, Any, Optional
from torch.ao.quantization.quantizer import (
from torch.ao.quantization import ObserverOrFakeQuantize
def _maybe_insert_input_observer_for_arg_or_kwarg(node: Union[Node, Any], arg: Argument, qconfig: QConfigAny, model: torch.nn.Module, named_modules: Dict[str, torch.nn.Module], obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize], is_qat: bool) -> Argument:
    """
    Given a `node` and an `arg`, inserts an input observer between
    `node` and `arg` if necessary.
    """
    if isinstance(arg, (list, tuple)):
        new_arg_to_return = []
        for inner_arg in arg:
            new_inner_arg = _maybe_insert_input_observer_for_arg_or_kwarg(node, inner_arg, qconfig, model, named_modules, obs_or_fq_map, is_qat)
            new_arg_to_return.append(new_inner_arg)
        return type(arg)(new_arg_to_return)
    if not isinstance(arg, Node):
        return arg
    assert isinstance(arg, Node)
    new_arg = arg
    original_arg = arg
    while _is_activation_post_process_node(original_arg, named_modules):
        original_arg = original_arg.args[0]
    assert isinstance(original_arg, Node), f'expect original argument to be a Node, but got: {type(original_arg)}'
    input_edge = (original_arg, node)
    if input_edge not in obs_or_fq_map:
        return new_arg
    input_edge_obs_or_fq = obs_or_fq_map[input_edge]
    if input_edge_obs_or_fq is None:
        return new_arg
    arg_as_output_obs_or_fq = obs_or_fq_map.get(original_arg, None)
    if arg_as_output_obs_or_fq is not None and id(arg_as_output_obs_or_fq) == id(input_edge_obs_or_fq):
        return new_arg
    existing_obs_node = None
    for maybe_obs_node in arg.users.keys():
        if not _is_activation_post_process_node(maybe_obs_node, named_modules):
            continue
        maybe_obs_mod = named_modules[maybe_obs_node.target]
        if id(maybe_obs_mod) == id(input_edge_obs_or_fq):
            return maybe_obs_node
    new_arg = _insert_obs_or_fq(arg, input_edge_obs_or_fq, model, named_modules, model.graph)
    return new_arg