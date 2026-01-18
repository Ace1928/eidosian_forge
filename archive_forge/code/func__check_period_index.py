from statsmodels.compat.numpy import NP_LT_2
import numpy as np
import pandas as pd
def _check_period_index(x, freq='M'):
    from pandas import DatetimeIndex, PeriodIndex
    if not isinstance(x.index, (DatetimeIndex, PeriodIndex)):
        raise ValueError('The index must be a DatetimeIndex or PeriodIndex')
    if x.index.freq is not None:
        inferred_freq = x.index.freqstr
    else:
        inferred_freq = pd.infer_freq(x.index)
    if not inferred_freq.startswith(freq):
        raise ValueError('Expected frequency {}. Got {}'.format(freq, inferred_freq))