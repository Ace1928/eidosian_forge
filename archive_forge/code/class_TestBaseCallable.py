import inspect
import unittest
from traits.api import (
class TestBaseCallable(unittest.TestCase):

    def test_override_validate(self):
        """ Verify `BaseCallable` can be subclassed to create new traits.
        """

        class ZeroArgsCallable(BaseCallable):

            def validate(self, object, name, value):
                if callable(value):
                    sig = inspect.signature(value)
                    if len(sig.parameters) == 0:
                        return value
                self.error(object, name, value)

        class Foo(HasTraits):
            value = ZeroArgsCallable
        Foo(value=lambda: 1)
        with self.assertRaises(TraitError):
            Foo(value=lambda x: x)
        with self.assertRaises(TraitError):
            Foo(value=1)

    def test_accepts_function(self):
        MyBaseCallable(value=lambda x: x)

    def test_accepts_method(self):
        MyBaseCallable(value=Dummy.instance_method)

    def test_accepts_type(self):
        MyBaseCallable(value=int)

    def test_accepts_none(self):
        MyBaseCallable(value=None)

    def test_rejects_non_callable(self):
        with self.assertRaises(TraitError):
            MyBaseCallable(value=Dummy())
        with self.assertRaises(TraitError):
            MyBaseCallable(value=1)