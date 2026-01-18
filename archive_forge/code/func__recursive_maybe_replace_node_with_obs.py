import copy
import torch
import warnings
from torch.fx import (
from torch.fx.graph import (
from torch.fx.node import Argument
from ..quantize import (
from ..observer import (
from ..qconfig import (
from ..qconfig_mapping import (
from .qconfig_mapping_utils import (
from .quantize_handler import (
from torch.ao.quantization import (
from torch.ao.quantization.utils import (
from ._equalize import (
from .pattern_utils import (
from .match_utils import (
from .utils import (
from torch.ao.quantization import (
from torch.ao.quantization.quantize import (
from ..utils import (
from ..backend_config.utils import (
from ..backend_config import (
from .custom_config import (
from torch.ao.quantization.quantizer import (
from torch.ao.quantization import ObserverOrFakeQuantize
from torch._subclasses import FakeTensor
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from dataclasses import asdict
def _recursive_maybe_replace_node_with_obs(maybe_node: Argument, model: torch.nn.Module, named_modules: Dict[str, torch.nn.Module], graph: Graph) -> Argument:
    """
        Navigate an arbitrary data structure of lists, tuples, dicts.
        For each container type, recurse on all inputs. Once any Node
        is found, insert an observer if needed and do not recurse further.

        For example, given a structure of

          {'foo1': [[bar1]], 'foo2': {'foo3': [[[bar3]]]}}

        we recurse down to bar1 and bar3, observe them if necessary,
        and if we inserted an observer then replace the original node
        with its observer.

        Returns the data structure with all nodes needing observation being
        replaced by their observers.
        """
    if isinstance(maybe_node, Node):
        arg_as_output_target_dtype = _get_arg_target_dtype_as_output(maybe_node, named_modules, obs_or_fq_map, is_qat)
        observer_mod = None
        arg_as_input_target_dtype = torch.float
        if 'target_dtype_info' in maybe_node.meta:
            observer_cls = maybe_node.meta['target_dtype_info'].get('input_act_obs_or_fq_ctr', None)
            if observer_cls is not None:
                observer_mod = observer_cls()
                arg_as_input_target_dtype = observer_mod.dtype
        need_obs = arg_as_output_target_dtype != arg_as_input_target_dtype and arg_as_input_target_dtype != torch.float
        if need_obs:
            assert observer_mod is not None
            observer_node = _insert_obs_or_fq(maybe_node, observer_mod, model, named_modules, graph)
            return observer_node
        else:
            return maybe_node
    elif isinstance(maybe_node, (list, tuple)):
        results = []
        for inner_node in maybe_node:
            results.append(_recursive_maybe_replace_node_with_obs(inner_node, model, named_modules, graph))
        if isinstance(maybe_node, list):
            return results
        else:
            return tuple(results)
    elif isinstance(maybe_node, dict):
        results_dict = {}
        for k, inner_v in maybe_node.items():
            results_dict[k] = _recursive_maybe_replace_node_with_obs(inner_v, model, named_modules, graph)
        return results_dict
    elif maybe_node is None:
        return None
    else:
        raise Exception('Unhandled type for returned node:', maybe_node)