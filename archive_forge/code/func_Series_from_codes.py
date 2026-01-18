import numpy as np
import six
from patsy import PatsyError
from patsy.util import (SortAnythingKey,
def Series_from_codes(codes, categories):
    c = pandas_Categorical_from_codes(codes, categories)
    return pandas.Series(c)