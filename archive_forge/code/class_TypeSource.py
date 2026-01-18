import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
@dataclasses.dataclass(frozen=True)
class TypeSource(ChainedSource):

    def __post_init__(self):
        assert self.base is not None

    def reconstruct(self, codegen):
        codegen.load_import_from('builtins', 'type')
        return self.base.reconstruct(codegen) + create_call_function(1, True)

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        return f'type({self.base.name()})'