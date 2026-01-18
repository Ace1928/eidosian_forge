import enum
import re
from typing import Optional, Union
import numpy as np
import pandas
from pandas.api.types import is_datetime64_dtype
class DlpackDeviceType(enum.IntEnum):
    """Integer enum for device type codes matching DLPack."""
    CPU = 1
    CUDA = 2
    CPU_PINNED = 3
    OPENCL = 4
    VULKAN = 7
    METAL = 8
    VPI = 9
    ROCM = 10