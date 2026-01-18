import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
@dataclasses.dataclass(frozen=True)
class FSDPNNModuleSource(NNModuleSource):

    def guard_source(self):
        return _GUARD_SOURCE_FSDP_MODULE[self.base.guard_source()]