import abc
import concurrent.futures
import contextlib
import inspect
import sys
import time
import traceback
from typing import List, Tuple
import pytest
import duet
import duet.impl as impl
class TestSync:

    def test_sync_on_overridden_method(self):

        class Foo:

            async def foo_async(self, a: int) -> int:
                return a * 2
            foo = duet.sync(foo_async)

        class Bar(Foo):

            async def foo_async(self, a: int) -> int:
                return a * 3
        assert Foo().foo(5) == 10
        assert Bar().foo(5) == 15

    def test_sync_on_abstract_method(self):

        class Foo(abc.ABC):

            @abc.abstractmethod
            async def foo_async(self, a: int) -> int:
                pass
            foo = duet.sync(foo_async)

        class Bar(Foo):

            async def foo_async(self, a: int) -> int:
                return a * 3
        with pytest.raises(TypeError, match="Can't instantiate abstract class Foo.*foo_async"):
            _ = Foo()
        assert Bar().foo(5) == 15

    def test_sync_on_classmethod(self):
        with pytest.raises(TypeError, match='duet.sync cannot be applied to classmethod'):

            class _Foo:

                @classmethod
                async def foo_async(cls, a: int) -> int:
                    return a * 2
                foo = duet.sync(foo_async)