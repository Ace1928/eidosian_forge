import itertools
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple, Union
from torchgen.model import (
from torchgen.utils import assert_never
class ScalarType(IntEnum):
    Byte = 0
    Char = 1
    Short = 2
    Int = 3
    Long = 4
    Float = 6
    Double = 7
    Bool = 11