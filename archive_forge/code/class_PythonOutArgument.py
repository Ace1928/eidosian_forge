from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
@dataclass(frozen=True)
class PythonOutArgument(PythonArgument):
    outputs: Tuple[PythonArgument, ...]

    @staticmethod
    def from_outputs(outputs: Tuple[PythonArgument, ...]) -> Optional['PythonOutArgument']:
        if not outputs:
            return None
        size = len(outputs)
        if size == 1:
            return PythonOutArgument(name=outputs[0].name, type=outputs[0].type, default='None', default_init=None, outputs=outputs)
        elif size > 1:
            if any((not a.type.is_tensor_like() for a in outputs)):
                raise RuntimeError(f'Unsupported output type: {outputs}')
            return PythonOutArgument(name='out', type=ListType(BaseType(BaseTy.Tensor), size), default='None', default_init=None, outputs=outputs)
        raise AssertionError('Unexpected PythonOutArgument size')