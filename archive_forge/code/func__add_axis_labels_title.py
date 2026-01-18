from __future__ import annotations
import math
import warnings
from collections import namedtuple
import numpy as np
from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
from scipy import optimize, special, interpolate, stats
from scipy._lib._bunch import _make_tuple_bunch
from scipy._lib._util import _rename_parameter, _contains_nan, _get_nan
from ._ansari_swilk_statistics import gscale, swilk
from . import _stats_py
from ._fit import FitResult
from ._stats_py import find_repeats, _normtest_finish, SignificanceResult
from .contingency import chi2_contingency
from . import distributions
from ._distn_infrastructure import rv_generic
from ._hypotests import _get_wilcoxon_distr
from ._axis_nan_policy import _axis_nan_policy_factory
def _add_axis_labels_title(plot, xlabel, ylabel, title):
    """Helper function to add axes labels and a title to stats plots."""
    try:
        if hasattr(plot, 'set_title'):
            plot.set_title(title)
            plot.set_xlabel(xlabel)
            plot.set_ylabel(ylabel)
        else:
            plot.title(title)
            plot.xlabel(xlabel)
            plot.ylabel(ylabel)
    except Exception:
        pass