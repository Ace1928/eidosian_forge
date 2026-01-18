import importlib
import logging
import os
import sys
import tempfile
from typing import Optional
import torch
from torch.distributed.nn.jit.templates.remote_module_template import (
def _do_instantiate_remote_module_template(generated_module_name, str_dict, enable_moving_cpu_tensors_to_cuda):
    generated_code_text = get_remote_module_template(enable_moving_cpu_tensors_to_cuda).format(**str_dict)
    out_path = os.path.join(INSTANTIATED_TEMPLATE_DIR_PATH, f'{generated_module_name}.py')
    _write(out_path, generated_code_text)
    importlib.invalidate_caches()
    generated_module = importlib.import_module(f'{generated_module_name}')
    return generated_module