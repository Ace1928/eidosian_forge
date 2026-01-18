from __future__ import print_function, division, absolute_import
import ctypes
import operator
from collections import OrderedDict
from math import ceil
from datashader import datashape
import numpy as np
from .internal_utils import IndexCallable, isidentifier
class CollectionPrinter:

    def __repr__(self):
        s = str(self)
        strs = ('"""%s"""' if '\n' in s else '"%s"') % s
        return 'dshape(%s)' % strs