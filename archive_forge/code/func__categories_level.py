from statsmodels.compat.python import lrange, lzip
from itertools import product
import numpy as np
from numpy import array, cumsum, iterable, r_
from pandas import DataFrame
from statsmodels.graphics import utils
def _categories_level(keys):
    """use the Ordered dict to implement a simple ordered set
    return each level of each category
    [[key_1_level_1,key_2_level_1],[key_1_level_2,key_2_level_2]]
    """
    res = []
    for i in zip(*keys):
        tuplefied = _tuplify(i)
        res.append(list({j: None for j in tuplefied}))
    return res