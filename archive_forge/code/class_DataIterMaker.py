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
class DataIterMaker(object):

    def __init__(self):
        self.i = -1

    def __call__(self):
        return self

    def __iter__(self):
        return self

    def next(self):
        self.i += 1
        if self.i > 1:
            raise StopIteration
        return self.i
    __next__ = next