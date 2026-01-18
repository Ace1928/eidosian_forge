from typing import Any, Dict, List, Optional, Set, Tuple, Union, Type, Callable
from torch.ao.quantization.quant_type import QuantType
import torch
import copy
import warnings
from torch.fx import (
from torch.fx.graph import (
from ..utils import (
from ..qconfig import (
from ..qconfig_mapping import QConfigMapping
from .qconfig_mapping_utils import (
from torch.ao.quantization.backend_config.utils import (
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.observer import _is_activation_post_process
from .graph_module import (
from ._equalize import update_obs_for_equalization, convert_eq_obs
from torch.nn.utils.parametrize import type_before_parametrizations
from .utils import (
from torch.ao.quantization.utils import (
from torch.ao.quantization.quantize import (
from torch.ao.quantization.stubs import DeQuantStub
from .custom_config import (
from .lower_to_fbgemm import lower_to_fbgemm
from ._decomposed import quantized_decomposed_lib  # noqa: F401
import operator
def _maybe_recursive_remove_dequantize(arg: Any, node: Node, graph: Graph) -> None:
    """ If the arg is a dequantize Node, or a list/tuple/dict of dequantize Node,
    we'll recursively remove the dequantize Node
    """
    if isinstance(arg, Node) and arg.op == 'call_method' and (arg.target == 'dequantize'):
        quantize_node = arg.args[0]
        node.replace_input_with(arg, quantize_node)
    elif isinstance(arg, (list, tuple)):
        for arg_element in arg:
            _maybe_recursive_remove_dequantize(arg_element, node, graph)
    elif isinstance(arg, dict):
        for arg_element in arg.values():
            _maybe_recursive_remove_dequantize(arg_element, node, graph)
    else:
        warnings.warn(f'Unsupported node type in recursive remove dequantize: {type(arg)}')