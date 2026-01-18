import traceback
import numpy as np
from scipy import sparse, spatial
from pygsp import utils
from pygsp.graphs import Graph  # prevent circular import in Python < 3.5
def _import_pfl():
    try:
        import pyflann as pfl
    except Exception:
        raise ImportError('Cannot import pyflann. Choose another nearest neighbors method or try to install it with pip (or conda) install pyflann (or pyflann3).')
    return pfl