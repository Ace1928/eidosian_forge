from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
@dataclass(frozen=True)
class DispatchLambdaArgumentExprs:
    exprs: Sequence[str]
    inits: Sequence[str]