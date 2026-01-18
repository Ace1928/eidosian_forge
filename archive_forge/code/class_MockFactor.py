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
class MockFactor(object):

    def __init__(self):
        from patsy.origin import Origin
        self.origin = Origin('MOCK', 1, 2)

    def eval(self, state, data):
        return state[data]

    def name(self):
        return 'MOCK MOCK'