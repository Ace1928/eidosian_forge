import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
def _to_ndimage_mode(mode):
    """Convert from `numpy.pad` mode name to the corresponding ndimage mode."""
    mode_translation_dict = dict(constant='constant', edge='nearest', symmetric='reflect', reflect='mirror', wrap='wrap')
    if mode not in mode_translation_dict:
        raise ValueError(f"Unknown mode: '{mode}', or cannot translate mode. The mode should be one of 'constant', 'edge', 'symmetric', 'reflect', or 'wrap'. See the documentation of numpy.pad for more info.")
    return _fix_ndimage_mode(mode_translation_dict[mode])