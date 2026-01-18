from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.model import (
from .types_base import Binding, CType, Expr
from torchgen.api import (
@dataclass(frozen=True)
class CppSignatureGroup:
    func: FunctionSchema
    signature: CppSignature
    faithful_signature: Optional[CppSignature]
    symint_signature: Optional[CppSignature]
    symint_faithful_signature: Optional[CppSignature]

    def most_faithful_signature(self) -> CppSignature:
        if self.faithful_signature:
            return self.faithful_signature
        else:
            return self.signature

    def signatures(self, *, symint: bool=True) -> Iterator[CppSignature]:
        yield self.signature
        if self.faithful_signature:
            yield self.faithful_signature
        if symint:
            if self.symint_signature:
                yield self.symint_signature
            if self.symint_faithful_signature:
                yield self.symint_faithful_signature

    @staticmethod
    def from_native_function(f: NativeFunction, *, method: bool, fallback_binding: bool=False) -> 'CppSignatureGroup':
        func = f.func

        def make_sig(*, faithful: bool, symint: bool) -> CppSignature:
            return CppSignature(func=func, faithful=faithful, symint=symint, method=method, fallback_binding=fallback_binding, cpp_no_default_args=f.cpp_no_default_args)

        def make_sigs(*, symint: bool) -> Tuple[CppSignature, Optional[CppSignature]]:
            faithful_signature: Optional[CppSignature] = None
            if func.arguments.tensor_options is not None or len(func.arguments.out) > 0:
                faithful_signature = make_sig(faithful=True, symint=symint)
            signature = make_sig(faithful=False, symint=symint)
            return (signature, faithful_signature)
        signature, faithful_signature = make_sigs(symint=False)
        symint_signature: Optional[CppSignature] = None
        symint_faithful_signature: Optional[CppSignature] = None
        if func.has_symint():
            symint_signature, symint_faithful_signature = make_sigs(symint=True)
        return CppSignatureGroup(func=func, signature=signature, faithful_signature=faithful_signature, symint_signature=symint_signature, symint_faithful_signature=symint_faithful_signature)