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
def _assign_default_kwargs(kws, call_func, source_func):
    """Assign default kwargs for call_func using values from source_func."""
    needed = inspect.signature(call_func).parameters
    defaults = inspect.signature(source_func).parameters
    for param in needed:
        if param in defaults and param not in kws:
            kws[param] = defaults[param].default
    return kws