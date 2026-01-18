import numpy as np
from numpy.linalg import eigvals, inv, solve, matrix_rank, pinv, svd
from scipy import stats
import pandas as pd
from patsy import DesignInfo
from statsmodels.compat.pandas import Substitution
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
@Substitution(hypotheses_doc=_hypotheses_doc)
def _multivariate_test(hypotheses, exog_names, endog_names, fn):
    """
    Multivariate linear model hypotheses testing

    For y = x * params, where y are the dependent variables and x are the
    independent variables, testing L * params * M = 0 where L is the contrast
    matrix for hypotheses testing and M is the transformation matrix for
    transforming the dependent variables in y.

    Algorithm:
        T = L*inv(X'X)*L'
        H = M'B'L'*inv(T)*LBM
        E =  M'(Y'Y - B'X'XB)M
    where H and E correspond to the numerator and denominator of a univariate
    F-test. Then find the eigenvalues of inv(H + E)*H from which the
    multivariate test statistics are calculated.

    .. [*] https://support.sas.com/documentation/cdl/en/statug/63033/HTML
           /default/viewer.htm#statug_introreg_sect012.htm

    Parameters
    ----------
    %(hypotheses_doc)s
    k_xvar : int
        The number of independent variables
    k_yvar : int
        The number of dependent variables
    fn : function
        a function fn(contrast_L, transform_M) that returns E, H, q, df_resid
        where q is the rank of T matrix

    Returns
    -------
    results : MANOVAResults
    """
    k_xvar = len(exog_names)
    k_yvar = len(endog_names)
    results = {}
    for hypo in hypotheses:
        if len(hypo) == 2:
            name, L = hypo
            M = None
            C = None
        elif len(hypo) == 3:
            name, L, M = hypo
            C = None
        elif len(hypo) == 4:
            name, L, M, C = hypo
        else:
            raise ValueError('hypotheses must be a tuple of length 2, 3 or 4. len(hypotheses)=%d' % len(hypo))
        if any((isinstance(j, str) for j in L)):
            L = DesignInfo(exog_names).linear_constraint(L).coefs
        else:
            if not isinstance(L, np.ndarray) or len(L.shape) != 2:
                raise ValueError('Contrast matrix L must be a 2-d array!')
            if L.shape[1] != k_xvar:
                raise ValueError('Contrast matrix L should have the same number of columns as exog! %d != %d' % (L.shape[1], k_xvar))
        if M is None:
            M = np.eye(k_yvar)
        elif any((isinstance(j, str) for j in M)):
            M = DesignInfo(endog_names).linear_constraint(M).coefs.T
        elif M is not None:
            if not isinstance(M, np.ndarray) or len(M.shape) != 2:
                raise ValueError('Transform matrix M must be a 2-d array!')
            if M.shape[0] != k_yvar:
                raise ValueError('Transform matrix M should have the same number of rows as the number of columns of endog! %d != %d' % (M.shape[0], k_yvar))
        if C is None:
            C = np.zeros([L.shape[0], M.shape[1]])
        elif not isinstance(C, np.ndarray):
            raise ValueError('Constant matrix C must be a 2-d array!')
        if C.shape[0] != L.shape[0]:
            raise ValueError('contrast L and constant C must have the same number of rows! %d!=%d' % (L.shape[0], C.shape[0]))
        if C.shape[1] != M.shape[1]:
            raise ValueError('transform M and constant C must have the same number of columns! %d!=%d' % (M.shape[1], C.shape[1]))
        E, H, q, df_resid = fn(L, M, C)
        EH = np.add(E, H)
        p = matrix_rank(EH)
        eigv2 = np.sort(eigvals(solve(EH, H)))
        stat_table = multivariate_stats(eigv2, p, q, df_resid)
        results[name] = {'stat': stat_table, 'contrast_L': L, 'transform_M': M, 'constant_C': C, 'E': E, 'H': H}
    return results