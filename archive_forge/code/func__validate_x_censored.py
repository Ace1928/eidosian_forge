import numpy as np
def _validate_x_censored(x, censored):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('`x` must be one-dimensional.')
    censored = np.asarray(censored)
    if censored.ndim != 1:
        raise ValueError('`censored` must be one-dimensional.')
    if (~np.isfinite(x)).any():
        raise ValueError('`x` must not contain nan or inf.')
    if censored.size != x.size:
        raise ValueError('`x` and `censored` must have the same length.')
    return (x, censored.astype(bool))