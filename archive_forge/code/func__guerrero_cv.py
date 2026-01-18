import numpy as np
from statsmodels.robust import mad
from scipy.optimize import minimize_scalar
def _guerrero_cv(self, x, bounds, window_length=4, scale='sd', options={'maxiter': 25}):
    """
        Computes lambda using guerrero's coefficient of variation. If no
        seasonality is present in the data, window_length is set to 4 (as
        per Guerrero and Perera, (2004)).

        NOTE: Seasonality-specific auxiliaries *should* provide their own
        seasonality parameter.

        Parameters
        ----------
        x : array_like
        bounds : tuple
            Numeric 2-tuple, that indicate the solution space for the lambda
            parameter.
        window_length : int
            Seasonality/grouping parameter. Default 4, as per Guerrero and
            Perera (2004). NOTE: this indicates the length of the individual
            groups, not the total number of groups!
        scale : {'sd', 'mad'}
            The dispersion measure to be used. 'sd' indicates the sample
            standard deviation, but the more robust 'mad' is also available.
        options : dict
            The options (as a dict) to be passed to the optimizer.
        """
    nobs = len(x)
    groups = int(nobs / window_length)
    grouped_data = np.reshape(x[nobs - groups * window_length:nobs], (groups, window_length))
    mean = np.mean(grouped_data, 1)
    scale = scale.lower()
    if scale == 'sd':
        dispersion = np.std(grouped_data, 1, ddof=1)
    elif scale == 'mad':
        dispersion = mad(grouped_data, axis=1)
    else:
        raise ValueError(f"Scale '{scale}' not understood.")

    def optim(lmbda):
        rat = np.divide(dispersion, np.power(mean, 1 - lmbda))
        return np.std(rat, ddof=1) / np.mean(rat)
    res = minimize_scalar(optim, bounds=bounds, method='bounded', options=options)
    return res.x