import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class EmptyDeclarationTests(unittest.TestCase):

    def _getEmpty(self):
        from zope.interface.declarations import Declaration
        return Declaration()

    def test___iter___empty(self):
        decl = self._getEmpty()
        self.assertEqual(list(decl), [])

    def test_flattened_empty(self):
        from zope.interface.interface import Interface
        decl = self._getEmpty()
        self.assertEqual(list(decl.flattened()), [Interface])

    def test___contains___empty(self):
        from zope.interface.interface import Interface
        decl = self._getEmpty()
        self.assertNotIn(Interface, decl)

    def test_extends_empty(self):
        from zope.interface.interface import Interface
        decl = self._getEmpty()
        self.assertTrue(decl.extends(Interface))
        self.assertTrue(decl.extends(Interface, strict=True))

    def test_interfaces_empty(self):
        decl = self._getEmpty()
        l = list(decl.interfaces())
        self.assertEqual(l, [])

    def test___sro___(self):
        from zope.interface.interface import Interface
        decl = self._getEmpty()
        self.assertEqual(decl.__sro__, (decl, Interface))

    def test___iro___(self):
        from zope.interface.interface import Interface
        decl = self._getEmpty()
        self.assertEqual(decl.__iro__, (Interface,))

    def test_get(self):
        decl = self._getEmpty()
        self.assertIsNone(decl.get('attr'))
        self.assertEqual(decl.get('abc', 'def'), 'def')
        self.assertFalse(decl._v_attrs)

    def test_changed_w_existing__v_attrs(self):
        decl = self._getEmpty()
        decl._v_attrs = object()
        decl.changed(decl)
        self.assertFalse(decl._v_attrs)