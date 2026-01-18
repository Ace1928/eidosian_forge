from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
def _omit_i(self, levels):
    if self.omit is None:
        return len(levels) - 1
    else:
        return _get_level(levels, self.omit)