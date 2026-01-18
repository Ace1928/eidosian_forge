import torch
from torch._C import _rename_privateuse1_backend, _get_privateuse1_backend_name
from typing import List, Optional, Union
def _check_register_once(module, attr):
    if hasattr(module, attr):
        raise RuntimeError(f'The custom device module of {module} has already been registered with {attr}')