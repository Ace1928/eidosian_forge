from numbers import Number
from statistics import NormalDist
import numpy as np
import pandas as pd
from .algorithms import bootstrap
from .utils import _check_argument
class EstimateAggregator:

    def __init__(self, estimator, errorbar=None, **boot_kws):
        """
        Data aggregator that produces an estimate and error bar interval.

        Parameters
        ----------
        estimator : callable or string
            Function (or method name) that maps a vector to a scalar.
        errorbar : string, (string, number) tuple, or callable
            Name of errorbar method (either "ci", "pi", "se", or "sd"), or a tuple
            with a method name and a level parameter, or a function that maps from a
            vector to a (min, max) interval, or None to hide errorbar. See the
            :doc:`errorbar tutorial </tutorial/error_bars>` for more information.
        boot_kws
            Additional keywords are passed to bootstrap when error_method is "ci".

        """
        self.estimator = estimator
        method, level = _validate_errorbar_arg(errorbar)
        self.error_method = method
        self.error_level = level
        self.boot_kws = boot_kws

    def __call__(self, data, var):
        """Aggregate over `var` column of `data` with estimate and error interval."""
        vals = data[var]
        if callable(self.estimator):
            estimate = self.estimator(vals)
        else:
            estimate = vals.agg(self.estimator)
        if self.error_method is None:
            err_min = err_max = np.nan
        elif len(data) <= 1:
            err_min = err_max = np.nan
        elif callable(self.error_method):
            err_min, err_max = self.error_method(vals)
        elif self.error_method == 'sd':
            half_interval = vals.std() * self.error_level
            err_min, err_max = (estimate - half_interval, estimate + half_interval)
        elif self.error_method == 'se':
            half_interval = vals.sem() * self.error_level
            err_min, err_max = (estimate - half_interval, estimate + half_interval)
        elif self.error_method == 'pi':
            err_min, err_max = _percentile_interval(vals, self.error_level)
        elif self.error_method == 'ci':
            units = data.get('units', None)
            boots = bootstrap(vals, units=units, func=self.estimator, **self.boot_kws)
            err_min, err_max = _percentile_interval(boots, self.error_level)
        return pd.Series({var: estimate, f'{var}min': err_min, f'{var}max': err_max})