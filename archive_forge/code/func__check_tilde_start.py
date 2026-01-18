import functools
import importlib
import importlib.resources
import re
import warnings
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
from .rcparams import rcParams
def _check_tilde_start(x):
    return bool(isinstance(x, str) and x.startswith('~'))