import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
@dataclasses.dataclass(frozen=True)
class TensorPropertySource(ChainedSource):
    prop: TensorProperty
    idx: Optional[int] = None

    def __post_init__(self):
        assert self.base is not None
        if self.prop is TensorProperty.STORAGE_OFFSET:
            assert self.idx is None
        else:
            assert self.idx is not None

    def reconstruct(self, codegen):
        instructions = [*self.base.reconstruct(codegen), codegen.create_load_attr(self.prop.method_name())]
        if self.idx is not None:
            instructions.append(codegen.create_load_const(self.idx))
        instructions.extend(create_call_function(1 if self.idx is not None else 0, True))
        return instructions

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        if self.prop is TensorProperty.SIZE:
            return f'{self.base.name()}.size()[{self.idx}]'
        elif self.prop is TensorProperty.STRIDE:
            return f'{self.base.name()}.stride()[{self.idx}]'
        elif self.prop is TensorProperty.STORAGE_OFFSET:
            assert self.idx is None
            return f'{self.base.name()}.storage_offset()'
        else:
            raise AssertionError(f'unhandled {self.prop}')