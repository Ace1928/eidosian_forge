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
def _maybe_propagate_dtype_for_node(node: Node, target_dtype: Union[torch.dtype, type], node_name_to_match_result_with_qconfig: Dict[str, _MatchResultWithQConfig]) -> None:
    """
    Assigns `target_dtype` to `node`, setting `is_dynamic` to False. If `node`
    is a general tensor shape op, also call this function recursively on
    the first argument, to propagate the dtype to the caller.
    """
    node.meta['target_dtype_info']['input_act_obs_or_fq_ctr'] = None
    node.meta['target_dtype_info']['output_act_obs_or_fq_ctr'] = None
    root_node, _, pattern, qhandler, qconfig = node_name_to_match_result_with_qconfig.get(node.name, (None, None, None, None, None))
    if qhandler is not None and qhandler.is_general_tensor_value_op():
        prev_node = node.args[0]
        if isinstance(prev_node, Node):
            _maybe_propagate_dtype_for_node(prev_node, target_dtype, node_name_to_match_result_with_qconfig)