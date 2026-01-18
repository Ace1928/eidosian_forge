import math
import numbers
import os
import cupy
from ._util import _get_inttype
def _get_block_size(check_warp_size=False):
    if check_warp_size:
        dev = cupy.cuda.runtime.getDevice()
        device_properties = cupy.cuda.runtime.getDeviceProperties(dev)
        return int(device_properties['warpSize'])
    else:
        return 32