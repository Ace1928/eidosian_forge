import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@dataclass(frozen=True)
class NativeFunctionsGroup:
    functional: NativeFunction
    inplace: Optional[NativeFunction]
    mutable: Optional[NativeFunction]
    out: NativeFunction

    @property
    def structured(self) -> bool:
        return self.out.structured

    def __post_init__(self) -> None:
        test_sig: FunctionSchema = self.functional.func.signature()
        for f in self.functions():
            if test_sig != f.func.signature():
                raise AssertionError(f"NativeFunctionsGroup constructed from two NativeFunctions that don't have matching signatures: {test_sig} != {f.func.signature()}")
            if self.structured != f.part_of_structured_group:
                raise AssertionError(f'NativeFunctionsGroup constructed from structured and unstructured functions: {self.out.func.name} and {f.func.name}')
        assert self.functional.func.kind() == SchemaKind.functional
        assert self.out.func.kind() == SchemaKind.out
        assert self.functional.namespace == self.out.namespace
        if self.inplace is not None:
            assert self.inplace.func.kind() == SchemaKind.inplace
            assert self.inplace.namespace == self.functional.namespace
        if self.mutable is not None:
            assert self.mutable.func.kind() == SchemaKind.mutable
            assert self.mutable.namespace == self.functional.namespace
            assert self.functional.func.name.name.functional_overload
        if self.structured:
            assert not self.out.has_composite_implicit_autograd_kernel and (not self.out.has_composite_implicit_autograd_nested_tensor_kernel)
            assert self.functional.structured_delegate == self.out.func.name, f'{self.functional.func.name} delegates to {self.functional.structured_delegate} but its actual delegate is {self.out.func.name}'
            if self.inplace is not None:
                assert self.inplace.structured_delegate == self.out.func.name
        generated_fns = sorted([str(f.func.name) for f in self.functions() if 'generated' in f.tags])
        generated_fns_str = ', '.join((str(x) for x in generated_fns))
        expected_generated_fns: Set[str] = set()
        for f in self.functions():
            expected_generated_fns.update((str(op) for op in f.autogen))
        expected_generated_fns_str = ', '.join((str(x) for x in sorted(expected_generated_fns)))
        if len(expected_generated_fns) == 0 and len(generated_fns) > 0:
            raise RuntimeError(f"The codegen expects to be able to generate '{generated_fns_str}'. In order to generate them however, we expect them to be called out explicitly in the yaml. Please add an 'autogen: {generated_fns_str}' line to the entry for {str(f.func.name)}")
        if expected_generated_fns_str != generated_fns_str:
            raise RuntimeError(f"The codegen expects to be able to generate '{generated_fns_str}'. To do so, it expects a line: 'autogen: {generated_fns_str}'. Instead, it found 'autogen: {expected_generated_fns_str}'")

    def signature(self) -> 'FunctionSchema':
        return self.out.func.signature()

    def functions(self) -> Iterator[NativeFunction]:
        yield self.functional
        yield self.out
        if self.inplace is not None:
            yield self.inplace
        if self.mutable is not None:
            yield self.mutable

    @property
    def root_name(self) -> str:
        return self.functional.root_name

    @staticmethod
    def from_dict(d: Dict[SchemaKind, NativeFunction]) -> Optional['NativeFunctionsGroup']:
        assert d
        if len(d) == 1:
            return None
        d = dict(d)
        functional = d.pop(SchemaKind.functional, None)
        inplace = d.pop(SchemaKind.inplace, None)
        mutable = d.pop(SchemaKind.mutable, None)
        out = d.pop(SchemaKind.out, None)
        assert not d
        assert functional is not None
        if out is None:
            return None
        return NativeFunctionsGroup(functional=functional, inplace=inplace, mutable=mutable, out=out)