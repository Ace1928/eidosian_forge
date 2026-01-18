from numbers import Number
from statistics import NormalDist
import numpy as np
import pandas as pd
from .algorithms import bootstrap
from .utils import _check_argument
class WeightedAggregator:

    def __init__(self, estimator, errorbar=None, **boot_kws):
        """
        Data aggregator that produces a weighted estimate and error bar interval.

        Parameters
        ----------
        estimator : string
            Function (or method name) that maps a vector to a scalar. Currently
            supports only "mean".
        errorbar : string or (string, number) tuple
            Name of errorbar method or a tuple with a method name and a level parameter.
            Currently the only supported method is "ci".
        boot_kws
            Additional keywords are passed to bootstrap when error_method is "ci".

        """
        if estimator != 'mean':
            raise ValueError(f"Weighted estimator must be 'mean', not {estimator!r}.")
        self.estimator = estimator
        method, level = _validate_errorbar_arg(errorbar)
        if method is not None and method != 'ci':
            raise ValueError(f"Error bar method must be 'ci', not {method!r}.")
        self.error_method = method
        self.error_level = level
        self.boot_kws = boot_kws

    def __call__(self, data, var):
        """Aggregate over `var` column of `data` with estimate and error interval."""
        vals = data[var]
        weights = data['weight']
        estimate = np.average(vals, weights=weights)
        if self.error_method == 'ci' and len(data) > 1:

            def error_func(x, w):
                return np.average(x, weights=w)
            boots = bootstrap(vals, weights, func=error_func, **self.boot_kws)
            err_min, err_max = _percentile_interval(boots, self.error_level)
        else:
            err_min = err_max = np.nan
        return pd.Series({var: estimate, f'{var}min': err_min, f'{var}max': err_max})