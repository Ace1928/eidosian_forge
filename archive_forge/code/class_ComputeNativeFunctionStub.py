from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from torchgen import dest
from torchgen.api.types import DispatcherSignature  # isort:skip
from torchgen.context import method_with_native_function
from torchgen.executorch.model import ETKernelIndex
from torchgen.model import DispatchKey, NativeFunction, Variant
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import concatMap, Target
@dataclass(frozen=True)
class ComputeNativeFunctionStub:

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if Variant.function not in f.variants:
            return None
        sig = DispatcherSignature.from_schema(f.func, prefix=f'wrapper_CPU_{f.func.name.overload_name}_', symint=False)
        assert sig is not None
        if len(f.func.returns) == 0:
            ret_name = ''
        elif len(f.func.returns) == 1:
            if f.func.arguments.out:
                ret_name = f.func.arguments.out[0].name
            else:
                ret_name = next((a.name for a in f.func.arguments.flat_non_out if a.type == f.func.returns[0].type), '')
            if not ret_name:
                raise Exception(f"Can't handle this return type {f.func}")
        else:
            assert len(f.func.arguments.out) == len(f.func.returns), f'Out variant number of returns need to match the number of out arguments. Got outs {str(f.func.arguments.out)} but returns {str(f.func.returns)}'
            tensor_type = 'at::Tensor &'
            comma = ', '
            ret_name = f'::std::tuple<{comma.join([tensor_type] * len(f.func.returns))}>(\n                {comma.join([r.name for r in f.func.arguments.out])}\n            )'
        ret_str = f'return {ret_name};' if len(f.func.returns) > 0 else ''
        return f'\n{sig.defn()} {{\n    {ret_str}\n}}\n    '