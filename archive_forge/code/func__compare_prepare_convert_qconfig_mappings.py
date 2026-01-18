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
def _compare_prepare_convert_qconfig_mappings(prepare_qconfig_mapping: QConfigMapping, convert_qconfig_mapping: QConfigMapping):
    """ Compare the qconfig_mapping passed in convert to the one from prepare and check the values

    Args:
      `prepare_qconfig_mapping`: configuration for prepare quantization step
      `convert_qconfig_mapping`: configuration for convert quantization step
    """
    assert qconfig_equals(prepare_qconfig_mapping.global_qconfig, convert_qconfig_mapping.global_qconfig), 'Expected global qconfigs to be the same in the prepare and convert quantization configs'
    prepare_dicts: List[OrderedDict] = [prepare_qconfig_mapping.object_type_qconfigs, prepare_qconfig_mapping.module_name_qconfigs, prepare_qconfig_mapping.module_name_regex_qconfigs]
    convert_dicts: List[OrderedDict] = [convert_qconfig_mapping.object_type_qconfigs, convert_qconfig_mapping.module_name_qconfigs, convert_qconfig_mapping.module_name_regex_qconfigs]
    dict_names = [_OBJECT_TYPE_DICT_KEY, _MODULE_NAME_DICT_KEY, _MODULE_NAME_REGEX_DICT_KEY]
    for i in range(len(prepare_dicts)):
        for name in prepare_dicts[i].keys():
            assert name in convert_dicts[i], f'Missing key {dict_names[i]} {name} in convert QConfigMapping                 when it was present in prepare'
            assert convert_dicts[i][name] is None or qconfig_equals(prepare_dicts[i][name], convert_dicts[i][name]), f'Expected convert QConfigMapping to have the same qconfig as prepare for key {dict_names[i]} {name};                 prepare: {prepare_dicts[i][name]}; convert: {convert_dicts[i][name]}'