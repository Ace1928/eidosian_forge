import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
@dataclasses.dataclass(frozen=True)
class GetItemSource(ChainedSource):
    index: Any
    index_is_slice: bool = False

    def __post_init__(self):
        assert self.base is not None
        if isinstance(self.index, slice):
            super().__setattr__('index', self.index.__reduce__())
            super().__setattr__('index_is_slice', True)

    def reconstruct(self, codegen):
        return [*reconstruct_getitem(self, codegen, index_is_slice=self.index_is_slice), create_instruction('BINARY_SUBSCR')]

    def guard_source(self):
        return self.base.guard_source()

    def unpack_slice(self):
        assert self.index_is_slice
        slice_class, slice_args = self.index
        return slice_class(*slice_args)

    def name(self):
        if isinstance(self.index, Source):
            return f'{self.base.name()}[{self.index.name()}]'
        elif self.index_is_slice:
            return f'{self.base.name()}[{self.unpack_slice()!r}]'
        elif isinstance(self.index, enum.Enum):
            return f'{self.base.name()}[{enum_repr(self.index, self.guard_source().is_local())}]'
        else:
            return f'{self.base.name()}[{self.index!r}]'