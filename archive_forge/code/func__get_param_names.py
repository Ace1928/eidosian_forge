from . import matrix
from . import utils
from builtins import super
from copy import copy as shallow_copy
from future.utils import with_metaclass
from inspect import signature
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import abc
import numbers
import numpy as np
import pickle
import pygsp
import sys
import tasklogger
import warnings
@classmethod
def _get_param_names(cls):
    """Get parameter names for the estimator"""
    init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
    if init is object.__init__:
        return []
    init_signature = signature(init)
    parameters = [p for p in init_signature.parameters.values() if p.name != 'self' and p.kind != p.VAR_KEYWORD]
    parameters = set([p.name for p in parameters])
    for superclass in cls.__bases__:
        try:
            parameters.update(superclass._get_param_names())
        except AttributeError:
            pass
    return parameters