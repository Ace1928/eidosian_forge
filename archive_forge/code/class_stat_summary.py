import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs, uniquecols
from ..doctools import document
from ..exceptions import PlotnineError
from .stat import stat
@document
class stat_summary(stat):
    """
    Calculate summary statistics depending on x

    {usage}

    Parameters
    ----------
    {common_parameters}
    fun_data : str | callable, default="mean_cl_boot"
        If string, it should be one of:

        ```python
        # Bootstrapped mean, confidence interval
        # Arguments:
        #     n_samples - No. of samples to draw
        #     confidence_interval
        #     random_state
        "mean_cl_boot"

        # Mean, C.I. assuming normal distribution
        # Arguments:
        #     confidence_interval
        "mean_cl_normal"

        # Mean, standard deviation * constant
        # Arguments:
        #     mult - multiplication factor
        "mean_sdl"

        # Median, outlier quantiles with equal tail areas
        # Arguments:
        #     confidence_interval
        "median_hilow"

        # Mean, Standard Errors * constant
        # Arguments:
        #     mult - multiplication factor
        "mean_se"
        ```

        or any function that takes a array and returns a dataframe
        with three columns named `y`, `ymin` and `ymax`.
    fun_y : callable, default=None
        Any function that takes a array_like and returns a value
    fun_ymin : callable, default=None
        Any function that takes an array_like and returns a value
    fun_ymax : callable, default=None
        Any function that takes an array_like and returns a value
    fun_args : dict, default=None
        Arguments to any of the functions. Provided the names of the
        arguments of the different functions are in not conflict, the
        arguments will be assigned to the right functions. If there is
        a conflict, create a wrapper function that resolves the
        ambiguity in the argument names.
    random_state : int | ~numpy.random.RandomState, default=None
        Seed or Random number generator to use. If `None`, then
        numpy global generator [](`numpy.random`) is used.

    Notes
    -----
    If any of `fun_y`, `fun_ymin` or `fun_ymax` are provided, the
    value of `fun_data` will be ignored.

    See Also
    --------
    plotnine.geom_pointrange
    """
    _aesthetics_doc = '\n    {aesthetics_table}\n\n    **Options for computed aesthetics**\n\n    ```python\n    "ymin"  # ymin computed by the summary function\n    "ymax"  # ymax computed by the summary function\n    "n"     # Number of observations at a position\n    ```\n\n    Calculated aesthetics are accessed using the `after_stat` function.\n    e.g. `after_stat(\'ymin\')`{.py}.\n    '
    REQUIRED_AES = {'x', 'y'}
    DEFAULT_PARAMS = {'geom': 'pointrange', 'position': 'identity', 'na_rm': False, 'fun_data': 'mean_cl_boot', 'fun_y': None, 'fun_ymin': None, 'fun_ymax': None, 'fun_args': None, 'random_state': None}
    CREATES = {'ymin', 'ymax', 'n'}

    def setup_params(self, data):
        keys = ('fun_data', 'fun_y', 'fun_ymin', 'fun_ymax')
        if not any((self.params[k] for k in keys)):
            raise PlotnineError('No summary function')
        if self.params['fun_args'] is None:
            self.params['fun_args'] = {}
        if 'random_state' not in self.params['fun_args'] and self.params['random_state']:
            random_state = self.params['random_state']
            if random_state is None:
                random_state = np.random
            elif isinstance(random_state, int):
                random_state = np.random.RandomState(random_state)
            self.params['fun_args']['random_state'] = random_state
        return self.params

    @classmethod
    def compute_panel(cls, data, scales, **params):
        func = make_summary_fun(params['fun_data'], params['fun_y'], params['fun_ymin'], params['fun_ymax'], params['fun_args'])
        summaries = []
        for (group, x), df in data.groupby(['group', 'x']):
            summary = func(df)
            summary['x'] = x
            summary['group'] = group
            summary['n'] = len(df)
            unique = uniquecols(df)
            if 'y' in unique:
                unique = unique.drop('y', axis=1)
            merged = summary.merge(unique, on=['group', 'x'])
            summaries.append(merged)
        new_data = pd.concat(summaries, axis=0, ignore_index=True)
        return new_data