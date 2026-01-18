from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def _callback_factory(self, exprs):
    return _Callback(self.indep, self.dep, self.params, exprs, Lambdify=self.be.Lambdify)