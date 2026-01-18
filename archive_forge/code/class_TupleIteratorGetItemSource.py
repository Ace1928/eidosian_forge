import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
@dataclasses.dataclass(frozen=True)
class TupleIteratorGetItemSource(GetItemSource):

    def reconstruct(self, codegen):
        codegen.load_import_from(utils.__name__, 'tuple_iterator_getitem')
        return [*self.base.reconstruct(codegen), codegen.create_load_const(self.index), *create_call_function(2, True)]

    def name(self):
        return f'___tuple_iterator_getitem({self.base.name()}, {self.index!r})'