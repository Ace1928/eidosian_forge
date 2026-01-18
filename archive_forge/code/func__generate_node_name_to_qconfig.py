import torch
import re
from collections import defaultdict, OrderedDict
from typing import Callable, Any, Dict, Tuple, Set, List, Union
from torch.ao.quantization import QConfig
from torch.ao.quantization.qconfig import _add_module_to_qconfig_obs_ctr, QConfigAny, qconfig_equals
from torch.ao.quantization.observer import (
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.backend_config.utils import (
from torch.fx import (
from torch.fx.graph import (
from torch.ao.nn.intrinsic import _FusedModule
from ..utils import (
from ..qconfig_mapping import (
def _generate_node_name_to_qconfig(root: torch.nn.Module, modules: Dict[str, torch.nn.Module], input_graph: Graph, qconfig_mapping: QConfigMapping, node_name_to_scope: Dict[str, Tuple[str, type]]) -> Dict[str, QConfigAny]:
    global_qconfig = qconfig_mapping.global_qconfig
    node_name_to_qconfig = {}
    submodule_to_object_type_to_cur_idx: Dict[str, Dict[Callable, int]] = defaultdict(lambda: defaultdict(int))
    for node in input_graph.nodes:
        qconfig = None
        if node.op == 'get_attr':
            module_name, _ = _parent_name(node.target)
            qconfig = _maybe_adjust_qconfig_for_module_type_or_name(qconfig_mapping, type(modules[module_name]), module_name, global_qconfig)
            qconfig_with_device_check = _add_module_to_qconfig_obs_ctr(qconfig, modules.get(node.target, None))
        elif node.op == 'call_function':
            function_qconfig = _get_object_type_qconfig(qconfig_mapping, node.target, global_qconfig)
            module_path, module_type = node_name_to_scope[node.name]
            qconfig = _maybe_adjust_qconfig_for_module_type_or_name(qconfig_mapping, module_type, module_path, function_qconfig)
            cur_object_type_idx = submodule_to_object_type_to_cur_idx[module_path][node.target]
            submodule_to_object_type_to_cur_idx[module_path][node.target] += 1
            qconfig = _maybe_adjust_qconfig_for_module_name_object_type_order(qconfig_mapping, module_path, node.target, cur_object_type_idx, qconfig)
            qconfig_with_device_check = _add_module_to_qconfig_obs_ctr(qconfig, modules.get(node.target, None))
        elif node.op == 'call_method':
            module_path, module_type = node_name_to_scope[node.name]
            qconfig = _maybe_adjust_qconfig_for_module_type_or_name(qconfig_mapping, node.target, module_path, global_qconfig)
            qconfig = _maybe_adjust_qconfig_for_module_type_or_name(qconfig_mapping, module_type, module_path, qconfig)
            qconfig_with_device_check = _add_module_to_qconfig_obs_ctr(qconfig, modules.get(node.target, None))
        elif node.op == 'call_module':
            if _is_activation_post_process(modules[node.target]):
                continue
            qconfig = _maybe_adjust_qconfig_for_module_type_or_name(qconfig_mapping, type(modules[node.target]), node.target, global_qconfig)
            module_path, module_type = node_name_to_scope[node.name]
            parent_name, _ = _parent_name(module_path)
            cur_object_type_idx = submodule_to_object_type_to_cur_idx[parent_name][module_type]
            submodule_to_object_type_to_cur_idx[parent_name][module_type] += 1
            qconfig = _maybe_adjust_qconfig_for_module_name_object_type_order(qconfig_mapping, parent_name, module_type, cur_object_type_idx, qconfig)
            qconfig_with_device_check = _add_module_to_qconfig_obs_ctr(qconfig, modules.get(node.target, None))
            modules[node.target].qconfig = qconfig_with_device_check
        else:
            qconfig_with_device_check = None
        node_name_to_qconfig[node.name] = qconfig_with_device_check
    return node_name_to_qconfig