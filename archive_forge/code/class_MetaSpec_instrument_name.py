import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
class MetaSpec_instrument_name(MetaSpec_track_name):
    type_byte = 4