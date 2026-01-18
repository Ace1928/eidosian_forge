from typing import Dict, Iterable, List, Union, cast
from ..compat import has_torch_amp, torch
from ..util import is_torch_array
def _tensors_per_device(self, tensors):
    tensors_per_device = dict()
    for tensor in tensors:
        device_tensors = tensors_per_device.setdefault(tensor.device, [])
        device_tensors.append(tensor)
    return tensors_per_device