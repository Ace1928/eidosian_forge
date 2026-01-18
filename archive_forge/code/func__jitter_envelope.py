import numpy as np
from scipy.stats import gaussian_kde
from . import utils
def _jitter_envelope(pos_data, xvals, violin, side):
    """Determine envelope for jitter markers."""
    if side == 'both':
        low, high = (-1.0, 1.0)
    elif side == 'right':
        low, high = (0, 1.0)
    elif side == 'left':
        low, high = (-1.0, 0)
    else:
        raise ValueError('`side` input incorrect: %s' % side)
    jitter_envelope = np.interp(pos_data, xvals, violin)
    jitter_coord = jitter_envelope * np.random.uniform(low=low, high=high, size=pos_data.size)
    return jitter_coord