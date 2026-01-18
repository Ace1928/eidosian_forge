from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.model import (
from .types_base import Binding, CType, Expr
from torchgen.api import (
@dataclass(frozen=True)
class ViewInverseSignature:
    g: NativeFunctionsViewGroup

    def name(self) -> str:
        assert self.g.view_copy is not None
        return functionalization.name(self.g, is_reverse=True, include_namespace=False)

    def decl(self) -> str:
        assert self.g.view_copy is not None
        return_type = functionalization.returns_type(self.g.view_copy.func)
        decls = [a.decl() for a in functionalization.inner_arguments(self.g.view_copy.func, is_reverse=True)]
        return f'static {return_type.cpp_type()} {self.name()}({', '.join(decls)});'