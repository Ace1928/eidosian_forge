from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
@dataclass(frozen=True)
class PythonArgument:
    name: str
    type: Type
    default: Optional[str]
    default_init: Optional[str]

    def argument_str(self, *, method: bool=False, symint: bool=True) -> str:
        type_str = argument_type_str(self.type, symint=symint).replace('const ', '').replace(' &', '')
        name = self.name
        if name == 'self' and type_str in ['Tensor', 'Number'] and (not method):
            name = 'input'
        if self.default is not None:
            default = {'nullptr': 'None', 'c10::nullopt': 'None', '{}': 'None'}.get(self.default, self.default)
            return f'{type_str} {name}={default}'
        else:
            return f'{type_str} {name}'

    def argument_str_pyi(self, *, method: bool=False, deprecated: bool=False) -> str:
        type_str = argument_type_str_pyi(self.type)
        name = self.name
        if name == 'self' and type_str == 'Tensor' and (not method) and (not deprecated):
            name = 'input'
        if name == 'from':
            name += '_'
        if name == 'out' and type_str == 'Tensor' and (not deprecated):
            type_str = 'Optional[' + type_str + ']'
        treat_as_no_default = deprecated and isinstance(self, PythonOutArgument) and (self.default == 'None')
        if self.default is not None and (not treat_as_no_default):
            if isinstance(self.type, ListType) and self.type.elem == BaseType(BaseTy.int) and self.default.startswith('{') and self.default.endswith('}'):
                default = '(' + self.default[1:-1] + ')'
            else:
                default = {'nullptr': 'None', 'c10::nullopt': 'None', '{}': 'None', 'MemoryFormat::Contiguous': 'contiguous_format', 'QScheme::PER_TENSOR_AFFINE': 'per_tensor_affine'}.get(self.default, self.default)
            return f'{name}: {type_str} = {default}'
        else:
            return f'{name}: {type_str}'