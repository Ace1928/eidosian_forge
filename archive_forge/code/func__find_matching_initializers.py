import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, Set, Tuple
import numpy as np
import onnx
from onnx import ModelProto, ValueInfoProto, numpy_helper
from ..utils import logging, recurse_getattr
def _find_matching_initializers(tied_params_with_op: List[Dict[str, str]], model: ModelProto, initializer_name_to_idx: Dict[str, int]):
    """
    From the torch parameter names in `tied_params`, find the matching initializers
    in the ONNX model.

    Args:
        tied_params_with_op (`List[Dict[str, str]]`):
            A list of groups of parameters that are tied, i.e. shared. For them,
            the torch module share the same pointer. The dictionary points to what type of nn.Module the parameter belongs to (e.g. `Linear`).
        model (`ModelProto`):
            The model in which the initializers should be looked for.
        initializer_name_to_idx (`Dict[str, int]`):
            A mapping from the model initializer name to their indices in model.graph.initializer, to ease the search.

    Returns:
        tied_groups_map (`Dict[Tuple[str], List[Dict[str, Any]]]`):
            A mapping from a tied weight group to the list of tied parameters torch name and potentially matching initializers (several in case it could not be exactly found).
    """
    tied_groups_map = {}
    for params in tied_params_with_op:
        torch_to_initializer = []
        for param_name, torch_op_name in params.items():
            identical_initializer = False
            if param_name in initializer_name_to_idx.keys():
                nodes_containing_initializer = set()
                for node in model.graph.node:
                    if param_name in node.input:
                        nodes_containing_initializer.add(node.name)
                torch_to_initializer.append({'param_name': param_name, 'initializer_name': {param_name}, 'nodes_containing_initializer': nodes_containing_initializer})
                identical_initializer = True
            if not identical_initializer:
                module_name = '/'.join(param_name.split('.')[:-1])
                if param_name.endswith('weight') and torch_op_name == 'Linear':
                    module_name += '/MatMul'
                elif param_name.endswith('bias') and torch_op_name == 'Linear':
                    module_name += '/Add'
                candidate_inputs = {}
                candidate_node_idxs = []
                for i, node in enumerate(model.graph.node):
                    if module_name in node.name:
                        candidate_node_idxs.append(i)
                for node_idx in candidate_node_idxs:
                    node_name = model.graph.node[node_idx].name
                    candidate_inputs[node_name] = list(model.graph.node[node_idx].input)
                torch_to_initializer_param = set()
                nodes_containing_initializer = set()
                for node_name, input_names in candidate_inputs.items():
                    for input_name in input_names:
                        if input_name in initializer_name_to_idx.keys():
                            torch_to_initializer_param.add(input_name)
                            nodes_containing_initializer.add(node_name)
                if len(torch_to_initializer_param) == 0:
                    logger.warning(f'Could not find ONNX initializer for torch parameter {param_name}. {param_name} will not be checked for deduplication.')
                torch_to_initializer.append({'param_name': param_name, 'initializer_name': torch_to_initializer_param, 'nodes_containing_initializer': nodes_containing_initializer})
        intersect = torch_to_initializer[0]['initializer_name']
        for i in range(1, len(params)):
            intersect = intersect.intersection(torch_to_initializer[i]['initializer_name'])
        if len(intersect) == 0:
            logger.warning('Found different candidate ONNX initializers (likely duplicate) for the tied weights:')
            not_found = []
            for i, torch_to_onnx_map in enumerate(torch_to_initializer):
                warn_string = f'\t{torch_to_onnx_map['param_name']}: {torch_to_onnx_map['initializer_name']}'
                if len(torch_to_onnx_map['initializer_name']) == 0:
                    not_found.append(i)
                    warn_string += ' --> ignored (may be a parameter from a part of the model not exported)'
                logger.warning(warn_string)
            for index in not_found[::-1]:
                del torch_to_initializer[index]
            if any((len(torch_to_onnx_map['initializer_name']) > 1 for torch_to_onnx_map in torch_to_initializer)):
                logger.warning(f'Could not find unique initializers corresponding to the torch tied parameters {params}. Deduplication will be skipped for this group of weights although it should be done. Please open an issue in Optimum repository.')
                continue
        tied_groups_map[tuple(params)] = torch_to_initializer
    return tied_groups_map