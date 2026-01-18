import functools
import hashlib
import logging
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, dviread
@classmethod
def _get_font_family_and_reduced(cls):
    """Return the font family name and whether the font is reduced."""
    ff = mpl.rcParams['font.family']
    ff_val = ff[0].lower() if len(ff) == 1 else None
    if len(ff) == 1 and ff_val in cls._font_families:
        return (ff_val, False)
    elif len(ff) == 1 and ff_val in cls._font_preambles:
        return (cls._font_types[ff_val], True)
    else:
        _log.info('font.family must be one of (%s) when text.usetex is True. serif will be used by default.', ', '.join(cls._font_families))
        return ('serif', False)