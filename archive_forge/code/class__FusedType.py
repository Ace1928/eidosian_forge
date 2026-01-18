from __future__ import absolute_import
import math, sys
class _FusedType(CythonType):
    __getitem__ = index_type