import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class Test_directlyProvides(unittest.TestCase):

    def _callFUT(self, *args, **kw):
        from zope.interface.declarations import directlyProvides
        return directlyProvides(*args, **kw)

    def test_w_normal_object(self):
        from zope.interface.declarations import ProvidesClass
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        class Foo:
            pass
        obj = Foo()
        self._callFUT(obj, IFoo)
        self.assertIsInstance(obj.__provides__, ProvidesClass)
        self.assertEqual(list(obj.__provides__), [IFoo])

    def test_w_class(self):
        from zope.interface.declarations import ClassProvides
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        class Foo:
            pass
        self._callFUT(Foo, IFoo)
        self.assertIsInstance(Foo.__provides__, ClassProvides)
        self.assertEqual(list(Foo.__provides__), [IFoo])

    def test_w_classless_object(self):
        from zope.interface.declarations import ProvidesClass
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        the_dict = {}

        class Foo:

            def __getattribute__(self, name):
                if name == '__class__':
                    return None
                raise NotImplementedError(name)

            def __setattr__(self, name, value):
                the_dict[name] = value
        obj = Foo()
        self._callFUT(obj, IFoo)
        self.assertIsInstance(the_dict['__provides__'], ProvidesClass)
        self.assertEqual(list(the_dict['__provides__']), [IFoo])