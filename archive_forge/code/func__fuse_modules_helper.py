import copy
import torch.nn as nn
from torch.ao.quantization.fuser_method_mappings import get_fuser_method
from torch.ao.quantization.fuser_method_mappings import fuse_conv_bn  # noqa: F401
from torch.ao.quantization.fuser_method_mappings import fuse_conv_bn_relu  # noqa: F401
from torch.nn.utils.parametrize import type_before_parametrizations
from typing import List, Optional
def _fuse_modules_helper(model, modules_to_fuse, is_qat, fuser_func=fuse_known_modules, fuse_custom_config_dict=None):
    if fuse_custom_config_dict is None:
        fuse_custom_config_dict = {}
    additional_fuser_method_mapping = fuse_custom_config_dict.get('additional_fuser_method_mapping', {})
    mod_list = []
    for item in modules_to_fuse:
        mod_list.append(_get_module(model, item))
    new_mod_list = fuser_func(mod_list, is_qat, additional_fuser_method_mapping)
    for i, item in enumerate(modules_to_fuse):
        _set_module(model, item, new_mod_list[i])