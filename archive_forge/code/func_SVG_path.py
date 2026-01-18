import itertools
import io
import base64
import numpy as np
import warnings
import matplotlib
from matplotlib.colors import colorConverter
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib import ticker
def SVG_path(path, transform=None, simplify=False):
    """Construct the vertices and SVG codes for the path

    Parameters
    ----------
    path : matplotlib.Path object

    transform : matplotlib transform (optional)
        if specified, the path will be transformed before computing the output.

    Returns
    -------
    vertices : array
        The shape (M, 2) array of vertices of the Path. Note that some Path
        codes require multiple vertices, so the length of these vertices may
        be longer than the list of path codes.
    path_codes : list
        A length N list of single-character path codes, N <= M. Each code is
        a single character, in ['L','M','S','C','Z']. See the standard SVG
        path specification for a description of these.
    """
    if transform is not None:
        path = path.transformed(transform)
    vc_tuples = [(vertices if path_code != Path.CLOSEPOLY else [], PATH_DICT[path_code]) for vertices, path_code in path.iter_segments(simplify=simplify)]
    if not vc_tuples:
        return (np.zeros((0, 2)), [])
    else:
        vertices, codes = zip(*vc_tuples)
        vertices = np.array(list(itertools.chain(*vertices))).reshape(-1, 2)
        return (vertices, list(codes))