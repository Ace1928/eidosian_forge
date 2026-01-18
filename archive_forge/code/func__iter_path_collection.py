import warnings
import itertools
from contextlib import contextmanager
from packaging.version import Version
import numpy as np
import matplotlib as mpl
from matplotlib import transforms
from .. import utils
@staticmethod
def _iter_path_collection(paths, path_transforms, offsets, styles):
    """Build an iterator over the elements of the path collection"""
    N = max(len(paths), len(offsets))
    if Version(mpl.__version__) < Version('1.4.0'):
        if path_transforms is None:
            path_transforms = [np.eye(3)]
    edgecolor = styles['edgecolor']
    if np.size(edgecolor) == 0:
        edgecolor = ['none']
    facecolor = styles['facecolor']
    if np.size(facecolor) == 0:
        facecolor = ['none']
    elements = [paths, path_transforms, offsets, edgecolor, styles['linewidth'], facecolor]
    it = itertools
    return it.islice(zip(*map(it.cycle, elements)), N)