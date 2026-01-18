from __future__ import absolute_import
import math, sys
class CythonType(CythonTypeObject):

    def _pointer(self, n=1):
        for i in range(n):
            self = pointer(self)
        return self