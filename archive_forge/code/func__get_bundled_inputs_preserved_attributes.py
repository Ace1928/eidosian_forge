import torch
from enum import Enum
from torch._C import _MobileOptimizerType as MobileOptimizerType
from typing import Optional, Set, List, AnyStr
def _get_bundled_inputs_preserved_attributes(script_module: torch.jit.ScriptModule, preserved_methods: List[str]) -> List[str]:
    bundled_inputs_attributes = []
    if hasattr(script_module, 'get_all_bundled_inputs'):
        bundled_inputs_attributes.append('get_all_bundled_inputs')
        bundled_inputs_attributes.append('get_num_bundled_inputs')
    if hasattr(script_module, 'get_bundled_inputs_functions_and_info'):
        bundled_inputs_attributes.append('get_bundled_inputs_functions_and_info')
        all_info = script_module.get_bundled_inputs_functions_and_info()
        for function_name in all_info:
            if function_name not in preserved_methods:
                bundled_inputs_attributes.append(function_name)
            bundled_inputs_attributes.append('get_all_bundled_inputs_for_' + function_name)
            bundled_inputs_attributes.append('_bundled_inputs_deflated_' + function_name)
    return bundled_inputs_attributes