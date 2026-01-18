from functools import wraps
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.tsatools import freq_to_period
def _get_pandas_wrapper(X, trim_head=None, trim_tail=None, names=None):
    index = X.index
    if trim_head is None and trim_tail is None:
        index = index
    elif trim_tail is None:
        index = index[trim_head:]
    elif trim_head is None:
        index = index[:-trim_tail]
    else:
        index = index[trim_head:-trim_tail]
    if hasattr(X, 'columns'):
        if names is None:
            names = X.columns
        return lambda x: X.__class__(x, index=index, columns=names)
    else:
        if names is None:
            names = X.name
        return lambda x: X.__class__(x, index=index, name=names)