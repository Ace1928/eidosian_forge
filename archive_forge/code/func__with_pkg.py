from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
@decorator
def _with_pkg(fun, pkg=None, min_version=None, *args, **kwargs):
    global __imported_pkgs
    if (pkg, min_version) not in __imported_pkgs:
        check_version(pkg, min_version=min_version)
        __imported_pkgs.add((pkg, min_version))
    return fun(*args, **kwargs)