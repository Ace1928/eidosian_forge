import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
class TorchScalarTypes(enum.Enum):
    QUINT8 = 13