import copyreg
import gc
import sys
import unittest
def _getRefcounts(self):
    copyreg.dispatch_table.clear()
    copyreg.dispatch_table.update(self._saved_pickle_registry)
    gc.collect()
    gc.collect()
    gc.collect()
    return sys.gettotalrefcount()