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
def _maybe_adjust_qconfig_for_module_name_object_type_order(qconfig_mapping: QConfigMapping, cur_module_path: str, cur_object_type: Callable, cur_object_type_idx: int, fallback_qconfig: QConfigAny) -> QConfigAny:
    for (module_name, object_type, index), qconfig in qconfig_mapping.module_name_object_type_order_qconfigs.items():
        if module_name == cur_module_path and object_type == cur_object_type and (index == cur_object_type_idx):
            return qconfig
    return fallback_qconfig