import unittest
class Test_ROComparison(unittest.TestCase):

    class MockC3:
        direct_inconsistency = False
        bases_had_inconsistency = False

    def _makeOne(self, c3=None, c3_ro=(), legacy_ro=()):
        from zope.interface.ro import _ROComparison
        return _ROComparison(c3 or self.MockC3(), c3_ro, legacy_ro)

    def test_inconsistent_label(self):
        comp = self._makeOne()
        self.assertEqual('no', comp._inconsistent_label)
        comp.c3.direct_inconsistency = True
        self.assertEqual('direct', comp._inconsistent_label)
        comp.c3.bases_had_inconsistency = True
        self.assertEqual('direct+bases', comp._inconsistent_label)
        comp.c3.direct_inconsistency = False
        self.assertEqual('bases', comp._inconsistent_label)