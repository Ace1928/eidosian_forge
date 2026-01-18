import copyreg
import gc
import sys
import unittest
class ReferenceLeakCheckerMixin(object):
    """A mixin class for TestCase, which checks reference counts."""
    NB_RUNS = 3

    def run(self, result=None):
        testMethod = getattr(self, self._testMethodName)
        expecting_failure_method = getattr(testMethod, '__unittest_expecting_failure__', False)
        expecting_failure_class = getattr(self, '__unittest_expecting_failure__', False)
        if expecting_failure_class or expecting_failure_method:
            return
        self._saved_pickle_registry = copyreg.dispatch_table.copy()
        super(ReferenceLeakCheckerMixin, self).run(result=result)
        super(ReferenceLeakCheckerMixin, self).run(result=result)
        oldrefcount = 0
        local_result = LocalTestResult(result)
        num_flakes = 0
        refcount_deltas = []
        while len(refcount_deltas) < self.NB_RUNS:
            oldrefcount = self._getRefcounts()
            super(ReferenceLeakCheckerMixin, self).run(result=local_result)
            newrefcount = self._getRefcounts()
            if newrefcount < oldrefcount and num_flakes < 2:
                num_flakes += 1
                continue
            num_flakes = 0
            refcount_deltas.append(newrefcount - oldrefcount)
        print(refcount_deltas, self)
        try:
            self.assertEqual(refcount_deltas, [0] * self.NB_RUNS)
        except Exception:
            result.addError(self, sys.exc_info())

    def _getRefcounts(self):
        copyreg.dispatch_table.clear()
        copyreg.dispatch_table.update(self._saved_pickle_registry)
        gc.collect()
        gc.collect()
        gc.collect()
        return sys.gettotalrefcount()