import itertools
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple, Union
from torchgen.model import (
from torchgen.utils import assert_never
@dataclass(frozen=True)
class ETKernelKeyOpArgMeta:
    arg_name: str
    dtype: str
    dim_order: Tuple[int, ...]

    def to_native_string(self) -> str:
        dtype_str = ScalarType[self.dtype].value
        dim_str = str(self.dim_order)[1:-1].replace(' ', '')
        return f'{dtype_str};{dim_str}'