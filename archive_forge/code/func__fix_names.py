import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
def _fix_names(field_spec):
    """ Replace names which are None with the next unused f%d name """
    names = field_spec['names']
    for i, name in enumerate(names):
        if name is not None:
            continue
        j = 0
        while True:
            name = f'f{j}'
            if name not in names:
                break
            j = j + 1
        names[i] = name