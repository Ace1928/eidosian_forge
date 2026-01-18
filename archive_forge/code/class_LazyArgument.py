from typing import Any, Dict, List, Optional, Tuple, Union
from torchgen.api.types import (
from torchgen.model import (
class LazyArgument:
    name: str
    orig_type: Type
    lazy_type_: Optional[CType]
    is_wrapped_scalar: bool
    is_generator: bool
    is_symint_or_list: bool
    symint: bool
    is_lazy_value: bool

    def __init__(self, arg: Argument, properties: 'LazyIrProperties', *, symint: bool):
        self.name = arg.name
        self.orig_type = arg.type
        self.symint = symint
        self.is_optional = isinstance(arg.type, OptionalType)
        self.is_generator = isGeneratorType(arg.type)
        self.lazy_type_ = process_ir_type(arg.type, properties, symint=symint)
        self.is_wrapped_scalar = isWrappedScalarType(arg.type)
        self.is_symint_or_list = symint and (isSymIntType(arg.type) or (isinstance(arg.type, OptionalType) and isSymIntType(arg.type.elem)))
        self.is_lazy_value = isValueType(self.lazy_type, properties)

    @property
    def lazy_type(self) -> CType:
        assert self.lazy_type_ is not None, f'Attempted to access lazy_type for invalid argument {self.name}'
        return self.lazy_type_