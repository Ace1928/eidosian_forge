import re
import os
import logging
from typing import Optional, List, Tuple
import ray._private.thirdparty.pynvml as pynvml
from ray._private.accelerators.accelerator import AcceleratorManager
@staticmethod
def _gpu_name_to_accelerator_type(name):
    if name is None:
        return None
    match = NVIDIA_GPU_NAME_PATTERN.match(name)
    return match.group(1) if match else None