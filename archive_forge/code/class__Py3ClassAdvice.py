import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class _Py3ClassAdvice:

    def _run_generated_code(self, code, globs, locs, fails_under_py3k=True):
        import warnings
        with warnings.catch_warnings(record=True) as log:
            warnings.resetwarnings()
            try:
                exec(code, globs, locs)
            except TypeError:
                return False
            else:
                if fails_under_py3k:
                    self.fail("Didn't raise TypeError")
            return None