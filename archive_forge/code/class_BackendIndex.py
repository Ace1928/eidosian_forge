import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@dataclass(frozen=True)
class BackendIndex:
    dispatch_key: DispatchKey
    use_out_as_primary: bool
    device_guard: bool
    external: bool
    index: Dict['OperatorName', BackendMetadata]

    @staticmethod
    def grow_index(parent_index: Dict[DispatchKey, Dict['OperatorName', BackendMetadata]], child_index: Dict[DispatchKey, Dict['OperatorName', BackendMetadata]]) -> None:
        for k, v in child_index.items():
            for op_name, metadata in v.items():
                assert op_name not in parent_index[k], f'duplicate operator {op_name} for dispatch key {k}'
                parent_index[k][op_name] = metadata

    def primary(self, g: NativeFunctionsGroup) -> NativeFunction:
        if self.use_out_as_primary:
            return g.out
        else:
            return g.functional

    def has_kernel(self, g: Union[NativeFunction, NativeFunctionsGroup]) -> bool:
        m = self.get_kernel(g)
        return m is not None

    def get_kernel(self, g: Union[NativeFunction, NativeFunctionsGroup]) -> Optional[BackendMetadata]:
        if isinstance(g, NativeFunction):
            f = g
        elif isinstance(g, NativeFunctionsGroup):
            f = self.primary(g)
        else:
            assert_never(g)
        if f.func.name not in self.index:
            return None
        return self.index[f.func.name]

    def native_function_class_name(self) -> Optional[str]:
        if self.external:
            return f'{str(self.dispatch_key)}NativeFunctions'
        else:
            return None