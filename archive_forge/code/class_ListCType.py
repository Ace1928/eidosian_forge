from dataclasses import dataclass
from typing import Dict
from torchgen.model import BaseTy, ScalarType
from .types_base import (
@dataclass(frozen=True)
class ListCType(CType):
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool=False) -> str:
        return f'c10::List<{self.elem.cpp_type()}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'c10::List<{self.elem.cpp_type_registration_declarations()}>'

    def remove_const_ref(self) -> 'CType':
        return ListCType(self.elem.remove_const_ref())