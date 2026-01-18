import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
@dataclasses.dataclass(frozen=True)
class NegateSource(ChainedSource):

    def __post_init__(self):
        assert self.base is not None

    def reconstruct(self, codegen):
        raise NotImplementedError()

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        return f'{self.base.name()}.__neg__()'