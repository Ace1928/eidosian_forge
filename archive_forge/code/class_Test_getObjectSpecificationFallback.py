import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class Test_getObjectSpecificationFallback(unittest.TestCase):

    def _getFallbackClass(self):
        from zope.interface.declarations import getObjectSpecificationFallback
        return getObjectSpecificationFallback
    _getTargetClass = _getFallbackClass

    def _callFUT(self, *args, **kw):
        return self._getTargetClass()(*args, **kw)

    def test_wo_existing_provides_classless(self):
        the_dict = {}

        class Foo:

            def __getattribute__(self, name):
                if name == '__class__':
                    raise AttributeError(name)
                try:
                    return the_dict[name]
                except KeyError:
                    raise AttributeError(name)

            def __setattr__(self, name, value):
                raise NotImplementedError()
        foo = Foo()
        spec = self._callFUT(foo)
        self.assertEqual(list(spec), [])

    def test_existing_provides_is_spec(self):
        from zope.interface.declarations import directlyProvides
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        def foo():
            raise NotImplementedError()
        directlyProvides(foo, IFoo)
        spec = self._callFUT(foo)
        self.assertIs(spec, foo.__provides__)

    def test_existing_provides_is_not_spec(self):

        def foo():
            raise NotImplementedError()
        foo.__provides__ = object()
        spec = self._callFUT(foo)
        self.assertEqual(list(spec), [])

    def test_existing_provides(self):
        from zope.interface.declarations import directlyProvides
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        class Foo:
            pass
        foo = Foo()
        directlyProvides(foo, IFoo)
        spec = self._callFUT(foo)
        self.assertEqual(list(spec), [IFoo])

    def test_wo_provides_on_class_w_implements(self):
        from zope.interface.declarations import implementer
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        @implementer(IFoo)
        class Foo:
            pass
        foo = Foo()
        spec = self._callFUT(foo)
        self.assertEqual(list(spec), [IFoo])

    def test_wo_provides_on_class_wo_implements(self):

        class Foo:
            pass
        foo = Foo()
        spec = self._callFUT(foo)
        self.assertEqual(list(spec), [])

    def test_catches_only_AttributeError_on_provides(self):
        MissingSomeAttrs.test_raises(self, self._callFUT, expected_missing='__provides__')

    def test_catches_only_AttributeError_on_class(self):
        MissingSomeAttrs.test_raises(self, self._callFUT, expected_missing='__class__', __provides__=None)

    def test_raises_AttributeError_when_provides_fails_type_check_AttributeError(self):

        class Foo:
            __provides__ = MissingSomeAttrs(AttributeError)
        self._callFUT(Foo())

    def test_raises_AttributeError_when_provides_fails_type_check_RuntimeError(self):

        class Foo:
            __provides__ = MissingSomeAttrs(RuntimeError)
        with self.assertRaises(RuntimeError) as exc:
            self._callFUT(Foo())
        self.assertEqual('__class__', exc.exception.args[0])