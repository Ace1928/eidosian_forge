import sys
from contextlib import contextmanager
from typing import (
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
class SpecialTyping:

    def custom_int_type(self) -> CustomIntType:
        return CustomIntType(1)

    def custom_heap_type(self) -> CustomHeapType:
        return CustomHeapType(HeapType())

    def int_type_alias(self) -> IntTypeAlias:
        return 1

    def heap_type_alias(self) -> HeapTypeAlias:
        return 1

    def literal(self) -> Literal[False]:
        return False

    def literal_string(self) -> LiteralString:
        return 'test'

    def self(self) -> Self:
        return self

    def any_str(self, x: AnyStr) -> AnyStr:
        return x

    def annotated(self) -> Annotated[float, 'positive number']:
        return 1

    def annotated_self(self) -> Annotated[Self, 'self with metadata']:
        self._metadata = 'test'
        return self

    def int_type_guard(self, x) -> TypeGuard[int]:
        return isinstance(x, int)

    def optional_float(self) -> Optional[float]:
        return 1.0

    def union_str_and_int(self) -> Union[str, int]:
        return ''

    def protocol(self) -> TestProtocol:
        return TestProtocolImplementer()

    def typed_dict(self) -> Movie:
        return {'name': 'The Matrix', 'year': 1999}