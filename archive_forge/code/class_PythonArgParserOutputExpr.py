from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
@dataclass(frozen=True)
class PythonArgParserOutputExpr:
    name: str
    expr: str
    index: int
    argument: PythonArgument

    @property
    def is_none_expr(self) -> str:
        return f'_r.isNone({self.index})'