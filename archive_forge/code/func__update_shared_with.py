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
def _update_shared_with(child: EdgeOrNode, qspec: QuantizationSpecBase, shared_with_map: Dict[EdgeOrNode, EdgeOrNode]):
    """Update the `shared_with_map` based on the qspec, this applies the `SharedQuantizationSpec`
    configuration and established the relationship between `edge_or_node` with the edge/node that it
    is pointing to, we'll use this information in the end to get the group id
    """
    if isinstance(qspec, SharedQuantizationSpec):
        parent = qspec.edge_or_node
        _union(parent, child, shared_with_map)