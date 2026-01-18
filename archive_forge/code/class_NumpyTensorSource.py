import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
@dataclasses.dataclass(frozen=True)
class NumpyTensorSource(ChainedSource):

    def name(self) -> str:
        return f'___from_numpy({self.base.name()})'

    def guard_source(self):
        return self.base.guard_source()

    def reconstruct(self, codegen):
        codegen.load_import_from('torch', 'as_tensor')
        return self.base.reconstruct(codegen) + create_call_function(1, True)