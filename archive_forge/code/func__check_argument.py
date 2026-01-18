import os
import inspect
import warnings
import colorsys
from contextlib import contextmanager
from urllib.request import urlopen, urlretrieve
from types import ModuleType
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
from matplotlib.cbook import normalize_kwargs
from seaborn._core.typing import deprecated
from seaborn.external.version import Version
from seaborn.external.appdirs import user_cache_dir
def _check_argument(param, options, value, prefix=False):
    """Raise if value for param is not in options."""
    if prefix and value is not None:
        failure = not any((value.startswith(p) for p in options if isinstance(p, str)))
    else:
        failure = value not in options
    if failure:
        raise ValueError(f'The value for `{param}` must be one of {options}, but {repr(value)} was passed.')
    return value