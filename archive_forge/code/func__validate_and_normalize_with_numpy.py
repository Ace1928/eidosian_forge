from html import escape as html_escape
from os.path import exists, isfile, splitext, abspath, join, isdir
from os import walk, sep, fsdecode
from IPython.core.display import DisplayObject, TextDisplayObject
from typing import Tuple, Iterable, Optional
@staticmethod
def _validate_and_normalize_with_numpy(data, normalize) -> Tuple[bytes, int]:
    import numpy as np
    data = np.array(data, dtype=float)
    if len(data.shape) == 1:
        nchan = 1
    elif len(data.shape) == 2:
        nchan = data.shape[0]
        data = data.T.ravel()
    else:
        raise ValueError('Array audio input must be a 1D or 2D array')
    max_abs_value = np.max(np.abs(data))
    normalization_factor = Audio._get_normalization_factor(max_abs_value, normalize)
    scaled = data / normalization_factor * 32767
    return (scaled.astype('<h').tobytes(), nchan)