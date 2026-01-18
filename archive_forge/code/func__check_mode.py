import warnings
import numpy
import cupy
def _check_mode(mode):
    if mode not in ('reflect', 'constant', 'nearest', 'mirror', 'wrap', 'grid-mirror', 'grid-wrap', 'grid-reflect'):
        msg = f'boundary mode not supported (actual: {mode})'
        raise RuntimeError(msg)
    return mode