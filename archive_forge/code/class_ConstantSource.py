import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
@dataclasses.dataclass(frozen=True)
class ConstantSource(Source):
    source_name: str

    def reconstruct(self, codegen):
        return [codegen.create_load_global(self.source_name, False, add=False)]

    def guard_source(self):
        return GuardSource.CONSTANT

    def name(self):
        return self.source_name

    def make_guard(self, fn):
        raise NotImplementedError()