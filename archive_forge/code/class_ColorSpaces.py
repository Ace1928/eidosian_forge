from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class ColorSpaces:
    DEVICE_RGB = '/DeviceRGB'
    DEVICE_CMYK = '/DeviceCMYK'
    DEVICE_GRAY = '/DeviceGray'