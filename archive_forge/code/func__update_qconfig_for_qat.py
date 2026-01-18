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
def _update_qconfig_for_qat(qconfig_mapping: QConfigMapping, backend_config: BackendConfig):
    """
    Update the qconfig_mapping to account for module swaps during QAT.
    During QAT we perform a module swap on the nn.Module types to the corresponding nn.qat.modules types.
    """
    module_to_qat_module_class = get_module_to_qat_module(backend_config)
    object_type_dict = qconfig_mapping.object_type_qconfigs
    new_object_type_dict = object_type_dict.copy()
    for k, v in new_object_type_dict.items():
        if k in module_to_qat_module_class:
            object_type_dict[module_to_qat_module_class[k]] = v