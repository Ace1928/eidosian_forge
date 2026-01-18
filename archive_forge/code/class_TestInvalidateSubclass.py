import platform
import time
import unittest
import pytest
from monty.functools import (
class TestInvalidateSubclass:

    def test_invalidate_attribute(self):
        called = []

        class Bar:

            @cached
            def bar(self):
                called.append('bar')
                return 1
        b = Bar()
        assert b.bar == 1
        assert len(called) == 1
        cached.invalidate(b, 'bar')
        assert b.bar == 1
        assert len(called) == 2

    def test_invalidate_attribute_twice(self):
        called = []

        class Bar:

            @cached
            def bar(self):
                called.append('bar')
                return 1
        b = Bar()
        assert b.bar == 1
        assert len(called) == 1
        cached.invalidate(b, 'bar')
        cached.invalidate(b, 'bar')
        assert b.bar == 1
        assert len(called) == 2

    def test_invalidate_uncalled_attribute(self):
        called = []

        class Bar:

            @cached
            def bar(self):
                called.append('bar')
                return 1
        b = Bar()
        assert len(called) == 0
        cached.invalidate(b, 'bar')

    def test_invalidate_private_attribute(self):
        called = []

        class Bar:

            @cached
            def __bar(self):
                called.append('bar')
                return 1

            def get_bar(self):
                return self.__bar
        b = Bar()
        assert b.get_bar() == 1
        assert len(called) == 1
        cached.invalidate(b, '__bar')
        assert b.get_bar() == 1
        assert len(called) == 2

    def test_invalidate_mangled_attribute(self):
        called = []

        class Bar:

            @cached
            def __bar(self):
                called.append('bar')
                return 1

            def get_bar(self):
                return self.__bar
        b = Bar()
        assert b.get_bar() == 1
        assert len(called) == 1
        cached.invalidate(b, '_Bar__bar')
        assert b.get_bar() == 1
        assert len(called) == 2

    def test_invalidate_reserved_attribute(self):
        called = []

        class Bar:

            @cached
            def __bar__(self):
                called.append('bar')
                return 1
        b = Bar()
        assert b.__bar__ == 1
        assert len(called) == 1
        cached.invalidate(b, '__bar__')
        assert b.__bar__ == 1
        assert len(called) == 2

    def test_invalidate_uncached_attribute(self):
        called = []

        class Bar:

            def bar(self):
                called.append('bar')
                return 1
        b = Bar()
        with pytest.raises(AttributeError, match="'Bar.bar' is not a cached attribute"):
            cached.invalidate(b, 'bar')

    def test_invalidate_uncached_private_attribute(self):
        called = []

        class Bar:

            def __bar(self):
                called.append('bar')
                return 1
        b = Bar()
        with pytest.raises(AttributeError, match="'Bar._Bar__bar' is not a cached attribute"):
            cached.invalidate(b, '__bar')

    def test_invalidate_unknown_attribute(self):
        called = []

        class Bar:

            @cached
            def bar(self):
                called.append('bar')
                return 1
        b = Bar()
        with pytest.raises(AttributeError, match="type object 'Bar' has no attribute 'baz'"):
            lazy_property.invalidate(b, 'baz')

    def test_invalidate_readonly_object(self):
        called = []

        class Bar:
            __slots__ = ()

            @cached
            def bar(self):
                called.append('bar')
                return 1
        b = Bar()
        with pytest.raises(AttributeError, match="'Bar' object has no attribute '__dict__'"):
            cached.invalidate(b, 'bar')

    def test_invalidate_superclass_attribute(self):
        called = []

        class Bar:

            @lazy_property
            def bar(self):
                called.append('bar')
                return 1
        b = Bar()
        with pytest.raises(AttributeError, match="'Bar.bar' is not a cached attribute"):
            cached.invalidate(b, 'bar')

    def test_invalidate_subclass_attribute(self):
        called = []

        class Bar:

            @cached
            def bar(self):
                called.append('bar')
                return 1
        b = Bar()
        assert b.bar == 1
        assert len(called) == 1
        lazy_property.invalidate(b, 'bar')
        assert b.bar == 1
        assert len(called) == 2