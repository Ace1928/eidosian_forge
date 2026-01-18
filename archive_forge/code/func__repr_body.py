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
def _repr_body(self):
    return ['Dimensions:'] + ['    {}: {}'.format(k, f'Unlimited (current: {self._dimensions[k].size})' if v is None else v) for k, v in self.dimensions.items()] + ['Groups:'] + [f'    {g}' for g in self.groups] + ['Variables:'] + [f'    {k}: {v.dimensions!r} {v.dtype}' for k, v in self.variables.items()] + ['Attributes:'] + [f'    {k}: {v!r}' for k, v in self.attrs.items()]