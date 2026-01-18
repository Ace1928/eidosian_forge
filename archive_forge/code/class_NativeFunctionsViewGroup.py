import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@dataclass(frozen=True)
class NativeFunctionsViewGroup:
    view: NativeFunction
    view_copy: Optional[NativeFunction]
    view_inplace: Optional[NativeFunction]

    def __post_init__(self) -> None:
        assert self.view.is_view_op
        if self.view_copy is None:
            assert not gets_generated_view_copy(self.view), f'{str(self.view.func.name)} appears to be a new operator that aliases its inputs. The codegen expects you to add a corresponding operator to native_functions.yaml: {get_view_copy_name(self.view)!s}. See Note [view_copy NativeFunctions] for details.'
        else:
            assert self.view_copy.func.name.name.base.endswith('_copy')
            assert self.view.func.signature() == self.view_copy.func.signature(strip_view_copy_name=True)
            assert 'view_copy' in self.view_copy.tags, f"{(str(self.view_copy.func.name), str(self.view.tags))} appears to be a view_copy operator. The codegen expects view_copy operators to be annotated with the 'view_copy' tag in native_functions.yaml. See Note [view_copy NativeFunction] for details."
        if self.view_inplace is not None:
            assert self.view.func.signature() == self.view_inplace.func.signature()
        if self.view.has_composite_implicit_autograd_kernel:
            if self.view_inplace is not None:
                assert self.view_inplace.has_composite_implicit_autograd_kernel, f'{str(self.view.func.name)} and {str(self.view_inplace.func.name)} must either both have CompositeImplicitAutograd kernels, or both not have composite kernels.'
        if self.view.has_composite_implicit_autograd_nested_tensor_kernel:
            if self.view_inplace is not None:
                assert self.view_inplace.has_composite_implicit_autograd_nested_tensor_kernel, f'{str(self.view.func.name)} and {str(self.view_inplace.func.name)} must either both have CompositeImplicitAutogradNestedTensor kernels, or both not have composite kernels.'

    def functions(self, *, include_copy: bool=True) -> Iterator[NativeFunction]:
        yield self.view
        if self.view_inplace is not None:
            yield self.view_inplace
        if self.view_copy is not None and include_copy:
            yield self.view_copy

    @property
    def root_name(self) -> str:
        return self.view.root_name

    @property
    def composite(self) -> bool:
        return self.view.has_composite_implicit_autograd_kernel