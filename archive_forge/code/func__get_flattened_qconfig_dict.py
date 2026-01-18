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
def _get_flattened_qconfig_dict(qconfig_mapping: QConfigMapping) -> Dict[Union[Callable, str], QConfigAny]:
    """ flatten the global, object_type and module_name qconfig
    to the same qconfig_dict so that it can be used by
    propagate_qconfig_ function.
    "module_name_regex" is ignored for now since it's not supported
    in propagate_qconfig_, but it can be fixed later.

    For example:
    Input: {
      "": qconfig,
      "object_type": [
        (torch.add, qconfig)
      ],
      "module_name": [
        ("conv", qconfig)
      ]
    }

    Output: {
      "": qconfig,
      torch.add: qconfig,
      "conv": qconfig
    }
    """
    flattened: Dict[Union[Callable, str], QConfigAny] = {'': qconfig_mapping.global_qconfig}
    for obj, qconfig in qconfig_mapping.object_type_qconfigs.items():
        flattened[obj] = qconfig
    for obj, qconfig in qconfig_mapping.module_name_qconfigs.items():
        flattened[obj] = qconfig
    return flattened