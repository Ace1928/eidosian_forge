import numpy as np
from ..Qt import QtCore, QtGui, QtWidgets
def _defersort(fn):

    def defersort(self, *args, **kwds):
        setSorting = False
        if self._sorting is None:
            self._sorting = self.isSortingEnabled()
            setSorting = True
            self.setSortingEnabled(False)
        try:
            return fn(self, *args, **kwds)
        finally:
            if setSorting:
                self.setSortingEnabled(self._sorting)
                self._sorting = None
    return defersort