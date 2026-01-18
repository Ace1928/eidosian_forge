import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
def _handle_missing(data, groups, formula, re_formula, vc_formula):
    tokens = set()
    forms = [formula]
    if re_formula is not None:
        forms.append(re_formula)
    if vc_formula is not None:
        forms.extend(vc_formula.values())
    from statsmodels.compat.python import asunicode
    from io import StringIO
    import tokenize
    skiptoks = {'(', ')', '*', ':', '+', '-', '**', '/'}
    for fml in forms:
        rl = StringIO(fml)

        def rlu():
            line = rl.readline()
            return asunicode(line, 'ascii')
        g = tokenize.generate_tokens(rlu)
        for tok in g:
            if tok not in skiptoks:
                tokens.add(tok.string)
    tokens = sorted(tokens & set(data.columns))
    data = data[tokens]
    ii = pd.notnull(data).all(1)
    if type(groups) is not str:
        ii &= pd.notnull(groups)
    return (data.loc[ii, :], groups[np.asarray(ii)])