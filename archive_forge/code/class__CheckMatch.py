import itertools
import six
import numpy as np
from patsy import PatsyError
from patsy.categorical import (guess_categorical,
from patsy.util import (atleast_2d_column_default,
from patsy.design_info import (DesignMatrix, DesignInfo,
from patsy.redundancy import pick_contrasts_for_term
from patsy.eval import EvalEnvironment
from patsy.contrasts import code_contrast_matrix, Treatment
from patsy.compat import OrderedDict
from patsy.missing import NAAction
class _CheckMatch(object):

    def __init__(self, name, eq_fn):
        self._name = name
        self._eq_fn = eq_fn
        self.value = None
        self._value_desc = None
        self._value_origin = None

    def check(self, seen_value, desc, origin):
        if self.value is None:
            self.value = seen_value
            self._value_desc = desc
            self._value_origin = origin
        elif not self._eq_fn(self.value, seen_value):
            msg = '%s mismatch between %s and %s' % (self._name, self._value_desc, desc)
            if isinstance(self.value, int):
                msg += ' (%r versus %r)' % (self.value, seen_value)
            raise PatsyError(msg, origin)