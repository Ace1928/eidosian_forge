from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
def _name_levels(prefix, levels):
    return ['[%s%s]' % (prefix, _obj_to_readable_str(level)) for level in levels]