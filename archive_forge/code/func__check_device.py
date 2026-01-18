import threading
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nccl_ops
def _check_device(tensor, expected=None):
    if not device.canonical_name(tensor.device):
        raise ValueError(f'Device assignment for tensor={tensor} required for nccl collective ops')
    if expected and expected != tensor.device:
        raise ValueError(f'Expected device {expected}, got {tensor.device} for tensor={tensor}.')