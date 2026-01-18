import os
import sys
import textwrap
import types
import re
import warnings
import functools
import platform
from .._utils import set_module
from numpy.core.numerictypes import issubclass_, issubsctype, issubdtype
from numpy.core import ndarray, ufunc, asarray
import numpy as np
def _opt_info():
    """
    Returns a string contains the supported CPU features by the current build.

    The string format can be explained as follows:
        - dispatched features that are supported by the running machine
          end with `*`.
        - dispatched features that are "not" supported by the running machine
          end with `?`.
        - remained features are representing the baseline.
    """
    from numpy.core._multiarray_umath import __cpu_features__, __cpu_baseline__, __cpu_dispatch__
    if len(__cpu_baseline__) == 0 and len(__cpu_dispatch__) == 0:
        return ''
    enabled_features = ' '.join(__cpu_baseline__)
    for feature in __cpu_dispatch__:
        if __cpu_features__[feature]:
            enabled_features += f' {feature}*'
        else:
            enabled_features += f' {feature}?'
    return enabled_features