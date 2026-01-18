import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
class MetaSpec_cue_marker(MetaSpec_text):
    type_byte = 7