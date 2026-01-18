import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class Test_noLongerProvides(unittest.TestCase):

    def _callFUT(self, *args, **kw):
        from zope.interface.declarations import noLongerProvides
        return noLongerProvides(*args, **kw)

    def test_wo_existing_provides(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        class Foo:
            pass
        obj = Foo()
        self._callFUT(obj, IFoo)
        self.assertEqual(list(obj.__provides__), [])

    def test_w_existing_provides_hit(self):
        from zope.interface.declarations import directlyProvides
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        class Foo:
            pass
        obj = Foo()
        directlyProvides(obj, IFoo)
        self._callFUT(obj, IFoo)
        self.assertEqual(list(obj.__provides__), [])

    def test_w_existing_provides_miss(self):
        from zope.interface.declarations import directlyProvides
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar')

        class Foo:
            pass
        obj = Foo()
        directlyProvides(obj, IFoo)
        self._callFUT(obj, IBar)
        self.assertEqual(list(obj.__provides__), [IFoo])

    def test_w_iface_implemented_by_class(self):
        from zope.interface.declarations import implementer
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        @implementer(IFoo)
        class Foo:
            pass
        obj = Foo()
        self.assertRaises(ValueError, self._callFUT, obj, IFoo)