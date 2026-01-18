from dataclasses import dataclass
from typing import Dict
from torchgen.api.types import (
from torchgen.model import BaseTy
@dataclass(frozen=True)
class OptionalCType(CType):
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool=False) -> str:
        return f'torch::executor::optional<{self.elem.cpp_type()}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'torch::executor::optional<{self.elem.cpp_type_registration_declarations()}>'

    def remove_const_ref(self) -> 'CType':
        return OptionalCType(self.elem.remove_const_ref())