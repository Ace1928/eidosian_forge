import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@dataclass(frozen=True)
class UfuncInnerLoop:
    name: str
    supported_dtypes: OrderedSet[ScalarType]
    ufunc_key: UfuncKey

    @staticmethod
    def parse(value: str, ufunc_key: UfuncKey) -> 'UfuncInnerLoop':
        name, supported_dtypes_str = value.split(' ', 1)
        assert supported_dtypes_str[0] == '('
        assert supported_dtypes_str[-1] == ')'
        supported_dtypes: OrderedSet[ScalarType] = OrderedSet()
        for k in supported_dtypes_str[1:-1].split(', '):
            supported_dtypes |= ScalarType.parse_set(k)
        return UfuncInnerLoop(name=name, supported_dtypes=supported_dtypes, ufunc_key=ufunc_key)