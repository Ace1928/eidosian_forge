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
@contextmanager
def _disable_autolayout():
    """Context manager for preventing rc-controlled auto-layout behavior."""
    orig_val = mpl.rcParams['figure.autolayout']
    try:
        mpl.rcParams['figure.autolayout'] = False
        yield
    finally:
        mpl.rcParams['figure.autolayout'] = orig_val