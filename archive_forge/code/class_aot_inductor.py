import os  # noqa: C101
import sys
from typing import Any, Dict, TYPE_CHECKING
import torch
from torch.utils._config_module import install_config_module
class aot_inductor:
    output_path = ''
    debug_compile = os.environ.get('AOT_INDUCTOR_DEBUG_COMPILE', '0') == '1'
    abi_compatible = is_fbcode()
    serialized_in_spec = ''
    serialized_out_spec = ''