import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def _python_lt(self, other_obj):
    try:
        return self.obj < other_obj
    except TypeError:
        return NotImplemented