from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
from torchgen.api import cpp, dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
@dataclass(frozen=True)
class GenCompositeViewCopyKernel:
    backend_index: BackendIndex

    @method_with_native_function
    def __call__(self, g: NativeFunctionsViewGroup) -> Optional[str]:
        if g.view_copy is None:
            return None
        metadata = self.backend_index.get_kernel(g.view_copy)
        assert metadata is not None
        if str(g.view_copy.func.name) == 'view_copy':
            assert metadata.kernel == 'view_copy_symint'
            return 'at::Tensor view_copy_symint(const at::Tensor & self, at::SymIntArrayRef size) {\n  c10::SymDimVector shape = infer_size_dv(size, self.sym_numel());\n  if (!at::detail::computeStride(self.sym_sizes(), self.sym_strides(), shape).has_value()) {\n    return self.reshape_symint(size);\n  } else {\n    auto output = at::_ops::view::call(self, size);\n    return output.clone(/*memory_format=*/at::MemoryFormat::Contiguous);\n  }\n}\n'
        view_copy_sig = NativeSignature(g.view_copy.func, symint=metadata.supports_symint())
        view_sig = DispatcherSignature(g.view.func)
        view_api_name = g.view.func.name.unambiguous_name()
        exprs = ', '.join([e.expr for e in translate(view_copy_sig.arguments(), view_sig.arguments())])
        assert len(g.view.func.returns) == 1
        assert g.view.func.returns[0].type == BaseType(BaseTy.Tensor) or g.view.func.returns[0].type == ListType(BaseType(BaseTy.Tensor), None)
        if g.view.func.returns[0].type == BaseType(BaseTy.Tensor):
            return_cloned_output = '  return output.clone(/*memory_format=*/at::MemoryFormat::Contiguous);'
        else:
            return_cloned_output = f'  {view_copy_sig.returns_type().cpp_type()} out_clone;\n  for (const auto i : c10::irange(output.size())) {{\n    out_clone.push_back(output[i].clone(/*memory_format=*/at::MemoryFormat::Contiguous));\n  }}\n  return out_clone;'
        return f'\n{view_copy_sig.defn(name=metadata.kernel)} {{\n  auto output = at::_ops::{view_api_name}::call({exprs});\n  {return_cloned_output}\n}}\n'