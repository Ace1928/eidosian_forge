import numpy as np
import warnings
def _structured_bootstrap(args, n_boot, units, func, func_kwargs, integers):
    """Resample units instead of datapoints."""
    unique_units = np.unique(units)
    n_units = len(unique_units)
    args = [[a[units == unit] for unit in unique_units] for a in args]
    boot_dist = []
    for i in range(int(n_boot)):
        resampler = integers(0, n_units, n_units, dtype=np.intp)
        sample = [[a[i] for i in resampler] for a in args]
        lengths = map(len, sample[0])
        resampler = [integers(0, n, n, dtype=np.intp) for n in lengths]
        sample = [[c.take(r, axis=0) for c, r in zip(a, resampler)] for a in sample]
        sample = list(map(np.concatenate, sample))
        boot_dist.append(func(*sample, **func_kwargs))
    return np.array(boot_dist)