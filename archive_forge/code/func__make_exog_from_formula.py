import pandas as pd
import patsy
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.compat.pandas import Appender
def _make_exog_from_formula(result, focus_var, summaries, values, num_points):
    """
    Create dataframes for exploring a fitted model as a function of one variable.

    This works for models fit with a formula.

    Returns
    -------
    dexog : data frame
        A data frame in which the focus variable varies and the other variables
        are fixed at specified or computed values.
    fexog : data frame
        The data frame `dexog` processed through the model formula.
    """
    model = result.model
    exog = model.data.frame
    if summaries is None:
        summaries = {}
    if values is None:
        values = {}
    if exog[focus_var].dtype is np.dtype('O'):
        raise ValueError('focus variable may not have object type')
    colnames = list(summaries.keys()) + list(values.keys()) + [focus_var]
    dtypes = [exog[x].dtype for x in colnames]
    varl = set(exog.columns.tolist()) - {model.endog_names}
    unmatched = varl - set(colnames)
    unmatched = list(unmatched)
    if len(unmatched) > 0:
        warnings.warn('%s in data frame but not in summaries or values.' % ', '.join(["'%s'" % x for x in unmatched]), ValueWarning)
    ix = range(num_points)
    fexog = pd.DataFrame(index=ix, columns=colnames)
    for d, x in zip(dtypes, colnames):
        fexog[x] = pd.Series(index=ix, dtype=d)
    pctls = np.linspace(0, 100, num_points).tolist()
    fvals = np.percentile(exog[focus_var], pctls)
    fvals = np.asarray(fvals)
    fexog.loc[:, focus_var] = fvals
    for ky in summaries.keys():
        fexog.loc[:, ky] = summaries[ky](exog.loc[:, ky])
    for ky in values.keys():
        fexog[ky] = values[ky]
    dexog = patsy.dmatrix(model.data.design_info, fexog, return_type='dataframe')
    return (dexog, fexog, fvals)