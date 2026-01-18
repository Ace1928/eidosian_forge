import os
import struct
from codecs import getincrementaldecoder, IncrementalDecoder
from enum import IntEnum
from typing import Generator, List, NamedTuple, Optional, Tuple, TYPE_CHECKING, Union
class RsvBits(NamedTuple):
    rsv1: bool
    rsv2: bool
    rsv3: bool