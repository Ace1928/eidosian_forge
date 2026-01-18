import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class AttributeTests(ElementTests):
    DEFAULT_NAME = 'TestAttribute'

    def _getTargetClass(self):
        from zope.interface.interface import Attribute
        return Attribute

    def test__repr__w_interface(self):
        method = self._makeOne()
        method.interface = type(self)
        r = repr(method)
        self.assertTrue(r.startswith('<zope.interface.interface.Attribute object at'), r)
        self.assertTrue(r.endswith(' ' + __name__ + '.AttributeTests.TestAttribute>'), r)

    def test__repr__wo_interface(self):
        method = self._makeOne()
        r = repr(method)
        self.assertTrue(r.startswith('<zope.interface.interface.Attribute object at'), r)
        self.assertTrue(r.endswith(' TestAttribute>'), r)

    def test__str__w_interface(self):
        method = self._makeOne()
        method.interface = type(self)
        r = str(method)
        self.assertEqual(r, __name__ + '.AttributeTests.TestAttribute')

    def test__str__wo_interface(self):
        method = self._makeOne()
        r = str(method)
        self.assertEqual(r, 'TestAttribute')