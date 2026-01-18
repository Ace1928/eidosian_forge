import os.path
import warnings
import weakref
from collections import ChainMap, Counter, OrderedDict, defaultdict
from collections.abc import Mapping
import h5py
import numpy as np
from packaging import version
from . import __version__
from .attrs import Attributes
from .dimensions import Dimension, Dimensions
from .utils import Frozen
def _unlabeled_dimension_mix(h5py_dataset):
    dimlist = getattr(h5py_dataset, 'dims', [])
    if not dimlist:
        status = 'nodim'
    else:
        dimset = set([len(j) for j in dimlist])
        if dimset ^ {0} == set():
            status = 'unlabeled'
        elif dimset & {0}:
            name = h5py_dataset.name.split('/')[-1]
            raise ValueError(f'malformed variable {name} has mixing of labeled and unlabeled dimensions.')
        else:
            status = 'labeled'
    return status