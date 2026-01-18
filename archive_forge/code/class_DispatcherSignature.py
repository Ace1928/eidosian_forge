from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.model import (
from .types_base import Binding, CType, Expr
from torchgen.api import (
@dataclass(frozen=True)
class DispatcherSignature:
    func: FunctionSchema
    prefix: str = ''
    symint: bool = True

    def arguments(self) -> List[Binding]:
        return dispatcher.arguments(self.func, symint=self.symint)

    def name(self) -> str:
        return self.prefix + dispatcher.name(self.func)

    def decl(self, name: Optional[str]=None) -> str:
        args_str = ', '.join((a.decl() for a in self.arguments()))
        if name is None:
            name = self.name()
        return f'{self.returns_type().cpp_type()} {name}({args_str})'

    def defn(self, name: Optional[str]=None, *, is_redispatching_fn: bool=False) -> str:
        args = [a.defn() for a in self.arguments()]
        if is_redispatching_fn:
            args = ['c10::DispatchKeySet dispatchKeySet'] + args
        args_str = ', '.join(args)
        if name is None:
            name = self.name()
        return f'{self.returns_type().cpp_type()} {name}({args_str})'

    def exprs(self) -> List[Expr]:
        return [Expr(a.name, a.nctype) for a in self.arguments()]

    def returns_type(self) -> CType:
        return dispatcher.returns_type(self.func.returns, symint=self.symint)

    def ptr_type(self) -> str:
        dispatcher_args_types_str = ', '.join((a.type for a in self.arguments()))
        return f'{self.returns_type().cpp_type()} (*)({dispatcher_args_types_str})'

    def type(self) -> str:
        dispatcher_args_types_str = ', '.join((a.type for a in self.arguments()))
        return f'{self.returns_type().cpp_type()} ({dispatcher_args_types_str})'

    @staticmethod
    def from_schema(func: FunctionSchema, *, prefix: str='', symint: bool=True) -> 'DispatcherSignature':
        return DispatcherSignature(func, prefix, symint)