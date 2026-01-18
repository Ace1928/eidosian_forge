import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class ProvidesClassStrictTests(ProvidesClassTests):

    def _getTargetClass(self):
        ProvidesClass = super()._getTargetClass()

        class StrictProvides(ProvidesClass):

            def _do_calculate_ro(self, base_mros):
                return ProvidesClass._do_calculate_ro(self, base_mros=base_mros, strict=True)
        return StrictProvides

    def test_overlapping_interfaces_corrected(self):
        from zope.interface import Interface
        from zope.interface import implementedBy
        from zope.interface import implementer

        class IBase(Interface):
            pass

        @implementer(IBase)
        class Base:
            pass
        spec = self._makeOne(Base, IBase)
        self.assertEqual(spec.__sro__, (spec, implementedBy(Base), IBase, implementedBy(object), Interface))