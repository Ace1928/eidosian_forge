import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class TestImmutableDeclaration(EmptyDeclarationTests):

    def _getTargetClass(self):
        from zope.interface.declarations import _ImmutableDeclaration
        return _ImmutableDeclaration

    def _getEmpty(self):
        from zope.interface.declarations import _empty
        return _empty

    def test_pickle(self):
        import pickle
        copied = pickle.loads(pickle.dumps(self._getEmpty()))
        self.assertIs(copied, self._getEmpty())

    def test_singleton(self):
        self.assertIs(self._getTargetClass()(), self._getEmpty())

    def test__bases__(self):
        self.assertEqual(self._getEmpty().__bases__, ())

    def test_change__bases__(self):
        empty = self._getEmpty()
        empty.__bases__ = ()
        self.assertEqual(self._getEmpty().__bases__, ())
        with self.assertRaises(TypeError):
            empty.__bases__ = (1,)

    def test_dependents(self):
        empty = self._getEmpty()
        deps = empty.dependents
        self.assertEqual({}, deps)
        deps[1] = 2
        self.assertEqual({}, empty.dependents)

    def test_changed(self):
        self._getEmpty().changed(None)

    def test_extends_always_false(self):
        self.assertFalse(self._getEmpty().extends(self))
        self.assertFalse(self._getEmpty().extends(self, strict=True))
        self.assertFalse(self._getEmpty().extends(self, strict=False))

    def test_get_always_default(self):
        self.assertIsNone(self._getEmpty().get('name'))
        self.assertEqual(self._getEmpty().get('name', 42), 42)

    def test_v_attrs(self):
        decl = self._getEmpty()
        self.assertEqual(decl._v_attrs, {})
        decl._v_attrs['attr'] = 42
        self.assertEqual(decl._v_attrs, {})
        self.assertIsNone(decl.get('attr'))
        attrs = decl._v_attrs = {}
        attrs['attr'] = 42
        self.assertEqual(decl._v_attrs, {})
        self.assertIsNone(decl.get('attr'))