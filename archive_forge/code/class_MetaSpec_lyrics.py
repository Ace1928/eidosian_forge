import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
class MetaSpec_lyrics(MetaSpec_text):
    type_byte = 5