import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class SpecificationBasePyTests(GenericSpecificationBaseTests):

    def test___call___miss(self):
        sb = self._makeOne()
        sb._implied = {}
        self.assertFalse(sb.isOrExtends(object()))

    def test___call___hit(self):
        sb = self._makeOne()
        testing = object()
        sb._implied = {testing: {}}
        self.assertTrue(sb(testing))

    def test_isOrExtends_miss(self):
        sb = self._makeOne()
        sb._implied = {}
        self.assertFalse(sb.isOrExtends(object()))

    def test_isOrExtends_hit(self):
        sb = self._makeOne()
        testing = object()
        sb._implied = {testing: {}}
        self.assertTrue(sb(testing))

    def test_implementedBy_hit(self):
        from zope.interface import interface
        sb = self._makeOne()

        class _Decl:
            _implied = {sb: {}}

        def _implementedBy(obj):
            return _Decl()
        with _Monkey(interface, implementedBy=_implementedBy):
            self.assertTrue(sb.implementedBy(object()))

    def test_providedBy_hit(self):
        from zope.interface import interface
        sb = self._makeOne()

        class _Decl:
            _implied = {sb: {}}

        def _providedBy(obj):
            return _Decl()
        with _Monkey(interface, providedBy=_providedBy):
            self.assertTrue(sb.providedBy(object()))