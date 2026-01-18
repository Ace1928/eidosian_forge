import numpy as np
def _validate_1d(a, name, allow_inf=False):
    if np.ndim(a) != 1:
        raise ValueError(f'`{name}` must be a one-dimensional sequence.')
    if np.isnan(a).any():
        raise ValueError(f'`{name}` must not contain nan.')
    if not allow_inf and np.isinf(a).any():
        raise ValueError(f'`{name}` must contain only finite values.')