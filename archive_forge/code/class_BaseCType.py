from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto, Enum
from typing import List, Optional, Union
from torchgen.model import Argument, SelfArgument, TensorOptionsArguments
@dataclass(frozen=True)
class BaseCType(CType):
    type: BaseCppType

    def cpp_type(self, *, strip_ref: bool=False) -> str:
        return str(self.type)

    def cpp_type_registration_declarations(self) -> str:
        return str(self.type).replace('at::', '')

    def remove_const_ref(self) -> 'CType':
        return self