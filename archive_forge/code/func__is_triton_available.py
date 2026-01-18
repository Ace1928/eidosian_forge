import logging
import os
import torch
from . import _cpp_lib
from .checkpoint import (  # noqa: E402, F401
@compute_once
def _is_triton_available():
    if not torch.cuda.is_available():
        return False
    if os.environ.get('XFORMERS_FORCE_DISABLE_TRITON', '0') == '1':
        return False
    if torch.cuda.get_device_capability('cuda') < (8, 0):
        return False
    try:
        from xformers.triton.softmax import softmax as triton_softmax
        return True
    except (ImportError, AttributeError):
        logger.warning('A matching Triton is not available, some optimizations will not be enabled', exc_info=True)
        return False