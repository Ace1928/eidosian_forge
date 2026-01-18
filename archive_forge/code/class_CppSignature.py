from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.model import (
from .types_base import Binding, CType, Expr
from torchgen.api import (
@dataclass(frozen=True)
class CppSignature:
    """
    A CppSignature represents a single overload in the C++ API.  For
    any given function schema, there may be multiple CppSignatures
    corresponding to it, based on how we desugar to C++.  See also
    CppSignatureGroup.
    """
    func: FunctionSchema
    method: bool
    faithful: bool
    symint: bool
    cpp_no_default_args: Set[str]
    fallback_binding: bool = False

    def arguments(self) -> Sequence[Binding]:
        return cpp.arguments(self.func.arguments, faithful=self.faithful, symint=self.symint, method=self.method, cpp_no_default_args=self.cpp_no_default_args)

    def name(self, *, suppress_symint_suffix: bool=False) -> str:
        n = cpp.name(self.func, faithful_name_for_out_overloads=self.faithful, symint_overload=False if suppress_symint_suffix else self.symint)
        if self.fallback_binding:
            n = f'__dispatch_{n}'
        return n

    def decl(self, *, name: Optional[str]=None, prefix: str='', is_redispatching_fn: bool=False, suppress_symint_suffix: bool=False) -> str:
        returns_type = cpp.returns_type(self.func.returns, symint=self.symint).cpp_type()
        cpp_args = [a.decl() for a in self.arguments()]
        if is_redispatching_fn:
            cpp_args = ['c10::DispatchKeySet dispatchKeySet'] + cpp_args
        cpp_args_str = ', '.join(cpp_args)
        if name is None:
            name = prefix + self.name(suppress_symint_suffix=suppress_symint_suffix)
        return f'{returns_type} {name}({cpp_args_str})'

    def defn(self, *, name: Optional[str]=None, prefix: str='', is_redispatching_fn: bool=False) -> str:
        returns_type = cpp.returns_type(self.func.returns, symint=self.symint).cpp_type()
        cpp_args = [a.defn() for a in self.arguments()]
        if is_redispatching_fn:
            cpp_args = ['c10::DispatchKeySet dispatchKeySet'] + cpp_args
        cpp_args_str = ', '.join(cpp_args)
        if name is None:
            name = prefix + self.name()
        return f'{returns_type} {name}({cpp_args_str})'

    def ptr_type(self) -> str:
        args_types_str = ', '.join((a.type for a in self.arguments()))
        return f'{cpp.returns_type(self.func.returns, symint=self.symint).cpp_type()} (*)({args_types_str})'

    def type(self) -> str:
        args_types_str = ', '.join((a.type for a in self.arguments()))
        return f'{cpp.returns_type(self.func.returns, symint=self.symint).cpp_type()} ({args_types_str})'