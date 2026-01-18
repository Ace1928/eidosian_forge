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
def _subterm_column_names_iter(factor_infos, subterm):
    total = 0
    for i, column_idxs in enumerate(_subterm_column_combinations(factor_infos, subterm)):
        name_pieces = []
        for factor, column_idx in zip(subterm.factors, column_idxs):
            fi = factor_infos[factor]
            if fi.type == 'numerical':
                if fi.num_columns > 1:
                    name_pieces.append('%s[%s]' % (factor.name(), column_idx))
                else:
                    assert column_idx == 0
                    name_pieces.append(factor.name())
            else:
                assert fi.type == 'categorical'
                contrast = subterm.contrast_matrices[factor]
                suffix = contrast.column_suffixes[column_idx]
                name_pieces.append('%s%s' % (factor.name(), suffix))
        if not name_pieces:
            yield 'Intercept'
        else:
            yield ':'.join(name_pieces)
        total += 1
    assert total == subterm.num_columns