from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def _Symbol(self, name, be=None):
    be = be or self.be
    try:
        return be.Symbol(name, real=True)
    except TypeError:
        return be.Symbol(name)