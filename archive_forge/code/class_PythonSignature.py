from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
@dataclass(frozen=True)
class PythonSignature:
    name: str
    input_args: Tuple[PythonArgument, ...]
    input_kwargs: Tuple[PythonArgument, ...]
    output_args: Optional[PythonOutArgument]
    returns: PythonReturns
    tensor_options_args: Tuple[PythonArgument, ...]
    method: bool

    @property
    def deprecated(self) -> bool:
        return False

    def arguments(self, *, skip_outputs: bool=False, skip_tensor_options: bool=False) -> Tuple[Union[PythonArgument, PythonOutArgument], ...]:
        result: List[Union[PythonArgument, PythonOutArgument]] = []
        result.extend(self.input_args)
        result.extend(self.input_kwargs)
        if self.output_args is not None and (not skip_outputs):
            result.append(self.output_args)
        if not skip_tensor_options:
            result.extend(self.tensor_options_args)
        return tuple(result)

    def arguments_count(self) -> int:
        return len(self.arguments())

    def output_idx(self) -> int:
        return len(self.input_args) + len(self.input_kwargs)

    def signature_str(self, *, skip_outputs: bool=False, symint: bool=True) -> str:
        args = self.arguments(skip_outputs=skip_outputs)
        schema_formals: List[str] = [a.argument_str(method=self.method, symint=symint) for a in args]
        positional_argc = len(self.input_args)
        if len(schema_formals) > positional_argc:
            schema_formals.insert(positional_argc, '*')
        return f'{self.name}({', '.join(schema_formals)})'

    def signature_str_pyi(self, *, skip_outputs: bool=False) -> str:
        args = self.arguments(skip_outputs=skip_outputs)
        schema_formals: List[str] = [a.argument_str_pyi(method=self.method) for a in args]
        positional_argc = len(self.input_args)
        if len(schema_formals) > positional_argc:
            schema_formals.insert(positional_argc, '*')
        returns_str = returns_str_pyi(self)
        if self.method:
            schema_formals.insert(0, 'self')
        return f'def {self.name}({', '.join(schema_formals)}) -> {returns_str}: ...'

    def signature_str_pyi_vararg(self, *, skip_outputs: bool=False) -> Optional[str]:
        args = self.arguments(skip_outputs=skip_outputs)
        schema_formals: List[str] = [a.argument_str_pyi(method=self.method) for a in args]
        num_args = self.arguments_count()
        num_positionalargs = len(self.input_args)
        have_vararg_version = False
        if num_args > 0:
            vararg_type = args[0].type
            if isinstance(vararg_type, ListType) and str(vararg_type.elem) in ['int', 'SymInt'] and (num_positionalargs == 1):
                have_vararg_version = True
        if not have_vararg_version:
            return None
        schema_formals[0] = '*' + args[0].name + ': _int'
        returns_str = returns_str_pyi(self)
        if self.method:
            schema_formals.insert(0, 'self')
        return f'def {self.name}({', '.join(schema_formals)}) -> {returns_str}: ...'