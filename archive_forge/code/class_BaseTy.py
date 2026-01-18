import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
class BaseTy(Enum):
    Generator = auto()
    ScalarType = auto()
    Tensor = auto()
    int = auto()
    Dimname = auto()
    DimVector = auto()
    float = auto()
    str = auto()
    bool = auto()
    Layout = auto()
    Device = auto()
    DeviceIndex = auto()
    Scalar = auto()
    MemoryFormat = auto()
    QScheme = auto()
    Storage = auto()
    Stream = auto()
    SymInt = auto()
    ConstQuantizerPtr = auto()