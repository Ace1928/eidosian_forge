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
def _subterm_column_combinations(factor_infos, subterm):
    columns_per_factor = []
    for factor in subterm.factors:
        if factor in subterm.contrast_matrices:
            columns = subterm.contrast_matrices[factor].matrix.shape[1]
        else:
            columns = factor_infos[factor].num_columns
        columns_per_factor.append(columns)
    return _column_combinations(columns_per_factor)