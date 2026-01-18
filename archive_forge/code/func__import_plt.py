from __future__ import division
import numpy as np
from pygsp import utils
def _import_plt():
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except Exception:
        raise ImportError('Cannot import matplotlib. Choose another backend or try to install it with pip (or conda) install matplotlib.')
    return plt