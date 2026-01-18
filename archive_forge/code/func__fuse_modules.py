import copy
import torch.nn as nn
from torch.ao.quantization.fuser_method_mappings import get_fuser_method
from torch.ao.quantization.fuser_method_mappings import fuse_conv_bn  # noqa: F401
from torch.ao.quantization.fuser_method_mappings import fuse_conv_bn_relu  # noqa: F401
from torch.nn.utils.parametrize import type_before_parametrizations
from typing import List, Optional
def _fuse_modules(model, modules_to_fuse, is_qat, inplace=False, fuser_func=fuse_known_modules, fuse_custom_config_dict=None):
    if not inplace:
        model = copy.deepcopy(model)
    if all((isinstance(module_element, str) for module_element in modules_to_fuse)):
        _fuse_modules_helper(model, modules_to_fuse, is_qat, fuser_func, fuse_custom_config_dict)
    else:
        for module_list in modules_to_fuse:
            _fuse_modules_helper(model, module_list, is_qat, fuser_func, fuse_custom_config_dict)
    return model