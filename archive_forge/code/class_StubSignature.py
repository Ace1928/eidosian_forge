from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.ufunc as ufunc
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.api.ufunc import UfunctorBindings
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import OrderedSet
@dataclass(frozen=True)
class StubSignature:
    g: NativeFunctionsGroup

    @property
    def name(self) -> str:
        return f'{str(self.g.functional.func.name.name)}_stub'

    @property
    def kernel_name(self) -> str:
        return f'{str(self.g.functional.func.name.name)}_kernel'

    @property
    def type_name(self) -> str:
        return f'{str(self.g.functional.func.name.name)}_fn'

    def arguments(self) -> List[Binding]:
        return ufunc.stub_arguments(self.g)

    def type(self) -> str:
        cpp_args = self.arguments()
        return f'void(*)(TensorIteratorBase&, {', '.join((a.type for a in cpp_args))})'

    def dispatch_decl(self) -> str:
        return f'DECLARE_DISPATCH({self.type_name}, {self.name})'

    def dispatch_defn(self) -> str:
        return f'DEFINE_DISPATCH({self.name})'

    def kernel_defn(self) -> str:
        return f'void {self.kernel_name}(TensorIteratorBase& iter, {', '.join((a.defn() for a in self.arguments()))})'

    def type_defn(self) -> str:
        return f'using {self.type_name} = {self.type()}'

    def call(self, ctx: Sequence[Binding]) -> str:
        return f'{self.name}(device_type(), *this, {', '.join((a.expr for a in translate(ctx, self.arguments())))})'

    def direct_call(self, ctx: Sequence[Binding]) -> str:
        return f'{self.kernel_name}(*this, {', '.join((a.expr for a in translate(ctx, self.arguments())))})'