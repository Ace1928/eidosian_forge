import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
@dataclasses.dataclass(frozen=True)
class RandomValueSource(Source):
    random_call_index: int

    def guard_source(self):
        return GuardSource.RANDOM_VALUE

    def reconstruct(self, codegen):
        return [codegen.create_load(codegen.tx.output.random_values_var), codegen.create_load_const(self.random_call_index), create_instruction('BINARY_SUBSCR')]

    def name(self):
        return f'random_value_{self.random_call_index}'