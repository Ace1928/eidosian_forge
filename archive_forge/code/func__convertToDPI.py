import io
import os
import re
import struct
from xml.etree import ElementTree
def _convertToDPI(density, unit):
    if unit == _UNIT_KM:
        return int(density * 2.54e-05 + 0.5)
    elif unit == _UNIT_100M:
        return int(density * 0.000254 + 0.5)
    elif unit == _UNIT_10M:
        return int(density * 0.00254 + 0.5)
    elif unit == _UNIT_1M:
        return int(density * 0.0254 + 0.5)
    elif unit == _UNIT_10CM:
        return int(density * 0.254 + 0.5)
    elif unit == _UNIT_CM:
        return int(density * 2.54 + 0.5)
    elif unit == _UNIT_MM:
        return int(density * 25.4 + 0.5)
    elif unit == _UNIT_0_1MM:
        return density * 254
    elif unit == _UNIT_0_01MM:
        return density * 2540
    elif unit == _UNIT_UM:
        return density * 25400
    return density