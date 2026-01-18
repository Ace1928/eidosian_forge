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
def _set_target_dtype_info_for_matched_node_pattern(matched_node_pattern: NodePattern, last_node: Node, qconfig: QConfigAny, qhandler: Optional[QuantizeHandler], backend_config: BackendConfig, named_modules: Dict[str, torch.nn.Module], cache_for_no_tensor_check: Dict[Node, bool], processed_nodes: Set[Node]) -> None:
    """ Sets the target_dtype_info for each node in matched_node_pattern
    Note: processed_nodes is used to ensure we only process each node once
    """
    if isinstance(matched_node_pattern, (list, tuple)):
        for node_pattern in matched_node_pattern:
            _set_target_dtype_info_for_matched_node_pattern(node_pattern, last_node, qconfig, qhandler, backend_config, named_modules, cache_for_no_tensor_check, processed_nodes)
    elif isinstance(matched_node_pattern, Node):
        assert isinstance(matched_node_pattern, Node)
        node = matched_node_pattern
        if node in processed_nodes:
            return
        processed_nodes.add(node)
        if qconfig is None:
            return
        target_dtype_info: Dict[str, Any] = _get_target_activation_dtype_for_node(node, qconfig, qhandler, named_modules, backend_config, cache_for_no_tensor_check)
        node.meta['target_dtype_info'] = target_dtype_info