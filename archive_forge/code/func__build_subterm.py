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
def _build_subterm(subterm, factor_infos, factor_values, out):
    assert subterm.num_columns == out.shape[1]
    out[...] = 1
    for i, column_idxs in enumerate(_subterm_column_combinations(factor_infos, subterm)):
        for factor, column_idx in zip(subterm.factors, column_idxs):
            if factor_infos[factor].type == 'categorical':
                contrast = subterm.contrast_matrices[factor]
                if np.any(factor_values[factor] < 0):
                    raise PatsyError("can't build a design matrix containing missing values", factor)
                out[:, i] *= contrast.matrix[factor_values[factor], column_idx]
            else:
                assert factor_infos[factor].type == 'numerical'
                assert factor_values[factor].shape[1] == factor_infos[factor].num_columns
                out[:, i] *= factor_values[factor][:, column_idx]