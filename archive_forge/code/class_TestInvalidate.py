import platform
import time
import unittest
import pytest
from monty.functools import (
class TestInvalidate:

    def test_invalidate_attribute(self):
        called = []

        class Foo:

            @lazy_property
            def foo(self):
                called.append('foo')
                return 1
        f = Foo()
        assert f.foo == 1
        assert len(called) == 1
        lazy_property.invalidate(f, 'foo')
        assert f.foo == 1
        assert len(called) == 2

    def test_invalidate_attribute_twice(self):
        called = []

        class Foo:

            @lazy_property
            def foo(self):
                called.append('foo')
                return 1
        f = Foo()
        assert f.foo == 1
        assert len(called) == 1
        lazy_property.invalidate(f, 'foo')
        lazy_property.invalidate(f, 'foo')
        assert f.foo == 1
        assert len(called) == 2

    def test_invalidate_uncalled_attribute(self):
        called = []

        class Foo:

            @lazy_property
            def foo(self):
                called.append('foo')
                return 1
        f = Foo()
        assert len(called) == 0
        lazy_property.invalidate(f, 'foo')

    def test_invalidate_private_attribute(self):
        called = []

        class Foo:

            @lazy_property
            def __foo(self):
                called.append('foo')
                return 1

            def get_foo(self):
                return self.__foo
        f = Foo()
        assert f.get_foo() == 1
        assert len(called) == 1
        lazy_property.invalidate(f, '__foo')
        assert f.get_foo() == 1
        assert len(called) == 2

    def test_invalidate_mangled_attribute(self):
        called = []

        class Foo:

            @lazy_property
            def __foo(self):
                called.append('foo')
                return 1

            def get_foo(self):
                return self.__foo
        f = Foo()
        assert f.get_foo() == 1
        assert len(called) == 1
        lazy_property.invalidate(f, '_Foo__foo')
        assert f.get_foo() == 1
        assert len(called) == 2

    def test_invalidate_reserved_attribute(self):
        called = []

        class Foo:

            @lazy_property
            def __foo__(self):
                called.append('foo')
                return 1
        f = Foo()
        assert f.__foo__ == 1
        assert len(called) == 1
        lazy_property.invalidate(f, '__foo__')
        assert f.__foo__ == 1
        assert len(called) == 2

    def test_invalidate_nonlazy_attribute(self):
        called = []

        class Foo:

            def foo(self):
                called.append('foo')
                return 1
        f = Foo()
        with pytest.raises(AttributeError, match="'Foo.foo' is not a lazy_property attribute"):
            lazy_property.invalidate(f, 'foo')

    def test_invalidate_nonlazy_private_attribute(self):
        called = []

        class Foo:

            def __foo(self):
                called.append('foo')
                return 1
        f = Foo()
        with pytest.raises(AttributeError, match="type object 'Foo' has no attribute 'foo'"):
            lazy_property.invalidate(f, 'foo')

    def test_invalidate_unknown_attribute(self):
        called = []

        class Foo:

            @lazy_property
            def foo(self):
                called.append('foo')
                return 1
        f = Foo()
        with pytest.raises(AttributeError, match="type object 'Foo' has no attribute 'bar'"):
            lazy_property.invalidate(f, 'bar')

    def test_invalidate_readonly_object(self):
        called = []

        class Foo:
            __slots__ = ()

            @lazy_property
            def foo(self):
                called.append('foo')
                return 1
        f = Foo()
        with pytest.raises(AttributeError, match="'Foo' object has no attribute '__dict__'"):
            lazy_property.invalidate(f, 'foo')