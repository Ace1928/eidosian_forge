from functools import partial
import numpy as np
from . import _catboost
def _param_to_string(dict_item):
    param, value = dict_item
    if param == 'misclass_cost_matrix':
        str_value = '/'.join(map(str, value.flatten()))
    else:
        str_value = str(value)
    return '{}={}'.format(param, str_value)