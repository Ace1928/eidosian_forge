from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto, Enum
from typing import List, Optional, Union
from torchgen.model import Argument, SelfArgument, TensorOptionsArguments
@dataclass(frozen=True)
class NamedCType:
    name: ArgName
    type: CType

    def cpp_type(self, *, strip_ref: bool=False) -> str:
        return self.type.cpp_type(strip_ref=strip_ref)

    def cpp_type_registration_declarations(self) -> str:
        return self.type.cpp_type_registration_declarations()

    def remove_const_ref(self) -> 'NamedCType':
        return NamedCType(self.name, self.type.remove_const_ref())

    def with_name(self, name: str) -> 'NamedCType':
        return NamedCType(name, self.type)