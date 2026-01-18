from __future__ import print_function
import six
import numpy as np
import pytest
from patsy import PatsyError
from patsy.util import (atleast_2d_column_default,
from patsy.desc import Term, INTERCEPT
from patsy.build import *
from patsy.categorical import C
from patsy.user_util import balanced, LookupFactor
from patsy.design_info import DesignMatrix, DesignInfo
class CheatingDataFrame(pandas.DataFrame):

    def __getitem__(self, key):
        if key == 'x':
            return pandas.DataFrame.__getitem__(self, key)[::-1]
        else:
            return pandas.DataFrame.__getitem__(self, key)