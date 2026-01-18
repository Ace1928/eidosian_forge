import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@dataclass(frozen=True)
class OperatorName:
    name: BaseOperatorName
    overload_name: str

    @staticmethod
    def parse(op_name: str) -> 'OperatorName':
        if '.' in op_name:
            name, overload_name = op_name.split('.', 1)
        else:
            name = op_name
            overload_name = ''
        r = OperatorName(name=BaseOperatorName.parse(name), overload_name=overload_name)
        assert str(r) == op_name, f'{str(r)} != {op_name}'
        return r

    def __str__(self) -> str:
        if self.overload_name:
            return f'{self.name}.{self.overload_name}'
        else:
            return f'{self.name}'

    def unambiguous_name(self) -> str:
        if self.overload_name:
            return f'{self.name}_{self.overload_name}'
        else:
            return f'{self.name}'

    def remove_inplace(self) -> 'OperatorName':
        return OperatorName(name=BaseOperatorName(base=self.name.base, inplace=False, dunder_method=self.name.dunder_method), overload_name=self.overload_name)

    def with_overload(self, overload: str) -> 'OperatorName':
        return OperatorName(name=BaseOperatorName(base=self.name.base, inplace=False, dunder_method=self.name.dunder_method), overload_name=overload)