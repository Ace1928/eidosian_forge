import os
from ...utils.filemanip import split_filename
from ..base import (
def _gen_model_options():
    """
        Generate all possible permutations of < multi - tensor > < single - tensor > options
        """
    single_tensor = ['dt', 'restore', 'algdt', 'nldt_pos', 'nldt', 'ldt_wtd']
    multi_tensor = ['cylcyl', 'cylcyl_eq', 'pospos', 'pospos_eq', 'poscyl', 'poscyl_eq', 'cylcylcyl', 'cylcylcyl_eq', 'pospospos', 'pospospos_eq', 'posposcyl', 'posposcyl_eq', 'poscylcyl', 'poscylcyl_eq']
    other = ['adc', 'ball_stick']
    model_list = single_tensor
    model_list.extend(other)
    model_list.extend([multi + ' ' + single for multi in multi_tensor for single in single_tensor])
    return model_list