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
def _examine_factor_types(factors, factor_states, data_iter_maker, NA_action):
    num_column_counts = {}
    cat_sniffers = {}
    examine_needed = set(factors)
    for data in data_iter_maker():
        for factor in list(examine_needed):
            value = factor.eval(factor_states[factor], data)
            if factor in cat_sniffers or guess_categorical(value):
                if factor not in cat_sniffers:
                    cat_sniffers[factor] = CategoricalSniffer(NA_action, factor.origin)
                done = cat_sniffers[factor].sniff(value)
                if done:
                    examine_needed.remove(factor)
            else:
                value = atleast_2d_column_default(value)
                _max_allowed_dim(2, value, factor)
                column_count = value.shape[1]
                num_column_counts[factor] = column_count
                examine_needed.remove(factor)
        if not examine_needed:
            break
    cat_levels_contrasts = {}
    for factor, sniffer in six.iteritems(cat_sniffers):
        cat_levels_contrasts[factor] = sniffer.levels_contrast()
    return (num_column_counts, cat_levels_contrasts)