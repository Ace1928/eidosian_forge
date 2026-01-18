from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.model import (
from .types_base import Binding, CType, Expr
from torchgen.api import (
@dataclass(frozen=True)
class StructuredImplSignature:
    g: NativeFunctionsGroup
    name: str

    def defn(self, name: Optional[str]=None) -> str:
        args_str = ', '.join((a.defn() for a in self.arguments()))
        return f'TORCH_IMPL_FUNC({self.name})({args_str})'

    def arguments(self) -> List[Binding]:
        return structured.impl_arguments(self.g)