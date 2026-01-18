import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class TestImplements(NameAndModuleComparisonTestsMixin, unittest.TestCase):

    def _getTargetClass(self):
        from zope.interface.declarations import Implements
        return Implements

    def _makeOne(self, *args, **kw):
        return self._getTargetClass()(*args, **kw)

    def _makeOneToCompare(self):
        from zope.interface.declarations import implementedBy

        class A:
            pass
        return implementedBy(A)

    def test_ctor_no_bases(self):
        impl = self._makeOne()
        self.assertEqual(impl.inherit, None)
        self.assertEqual(impl.declared, ())
        self.assertEqual(impl.__name__, '?')
        self.assertEqual(list(impl.__bases__), [])

    def test___repr__(self):
        impl = self._makeOne()
        impl.__name__ = 'Testing'
        self.assertEqual(repr(impl), 'classImplements(Testing)')

    def test___reduce__(self):
        from zope.interface.declarations import implementedBy
        impl = self._makeOne()
        self.assertEqual(impl.__reduce__(), (implementedBy, (None,)))

    def test_sort(self):
        from zope.interface.declarations import implementedBy

        class A:
            pass

        class B:
            pass
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        self.assertEqual(implementedBy(A), implementedBy(A))
        self.assertEqual(hash(implementedBy(A)), hash(implementedBy(A)))
        self.assertTrue(implementedBy(A) < None)
        self.assertTrue(None > implementedBy(A))
        self.assertTrue(implementedBy(A) < implementedBy(B))
        self.assertTrue(implementedBy(A) > IFoo)
        self.assertTrue(implementedBy(A) <= implementedBy(B))
        self.assertTrue(implementedBy(A) >= IFoo)
        self.assertTrue(implementedBy(A) != IFoo)

    def test_proxy_equality(self):

        class Proxy:

            def __init__(self, wrapped):
                self._wrapped = wrapped

            def __getattr__(self, name):
                raise NotImplementedError()

            def __eq__(self, other):
                return self._wrapped == other

            def __ne__(self, other):
                return self._wrapped != other
        from zope.interface.declarations import implementedBy

        class A:
            pass

        class B:
            pass
        implementedByA = implementedBy(A)
        implementedByB = implementedBy(B)
        proxy = Proxy(implementedByA)
        self.assertTrue(implementedByA == implementedByA)
        self.assertTrue(implementedByA != implementedByB)
        self.assertTrue(implementedByB != implementedByA)
        self.assertTrue(proxy == implementedByA)
        self.assertTrue(implementedByA == proxy)
        self.assertFalse(proxy != implementedByA)
        self.assertFalse(implementedByA != proxy)
        self.assertTrue(proxy != implementedByB)
        self.assertTrue(implementedByB != proxy)

    def test_changed_deletes_super_cache(self):
        impl = self._makeOne()
        self.assertIsNone(impl._super_cache)
        self.assertNotIn('_super_cache', impl.__dict__)
        impl._super_cache = 42
        self.assertIn('_super_cache', impl.__dict__)
        impl.changed(None)
        self.assertIsNone(impl._super_cache)
        self.assertNotIn('_super_cache', impl.__dict__)

    def test_changed_does_not_add_super_cache(self):
        impl = self._makeOne()
        self.assertIsNone(impl._super_cache)
        self.assertNotIn('_super_cache', impl.__dict__)
        impl.changed(None)
        self.assertIsNone(impl._super_cache)
        self.assertNotIn('_super_cache', impl.__dict__)