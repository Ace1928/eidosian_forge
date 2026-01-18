import abc
import sys
import traceback
import warnings
from io import StringIO
from decorator import decorator
from traitlets.config.configurable import Configurable
from .getipython import get_ipython
from ..utils.sentinel import Sentinel
from ..utils.dir2 import get_real_method
from ..lib import pretty
from traitlets import (
from typing import Any
@default('type_printers')
def _type_printers_default(self):
    d = pretty._type_pprinters.copy()
    d[float] = lambda obj, p, cycle: p.text(self.float_format % obj)
    if 'numpy' in sys.modules:
        import numpy
        d[numpy.float64] = lambda obj, p, cycle: p.text(self.float_format % obj)
    return d