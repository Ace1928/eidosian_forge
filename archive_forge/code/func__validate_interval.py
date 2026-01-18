import numpy as np
def _validate_interval(interval):
    interval = np.asarray(interval)
    if interval.shape == (0,):
        interval = interval.reshape((0, 2))
    if interval.ndim != 2 or interval.shape[-1] != 2:
        raise ValueError(f'`interval` must be a two-dimensional array with shape (m, 2), where m is the number of interval-censored values, but got shape {interval.shape}')
    if np.isnan(interval).any():
        raise ValueError('`interval` must not contain nan.')
    if np.isinf(interval).all(axis=1).any():
        raise ValueError('In each row in `interval`, both values must not be infinite.')
    if (interval[:, 0] > interval[:, 1]).any():
        raise ValueError('In each row of `interval`, the left value must not exceed the right value.')
    uncensored_mask = interval[:, 0] == interval[:, 1]
    left_mask = np.isinf(interval[:, 0])
    right_mask = np.isinf(interval[:, 1])
    interval_mask = np.isfinite(interval).all(axis=1) & ~uncensored_mask
    uncensored2 = interval[uncensored_mask, 0]
    left2 = interval[left_mask, 1]
    right2 = interval[right_mask, 0]
    interval2 = interval[interval_mask]
    return (uncensored2, left2, right2, interval2)