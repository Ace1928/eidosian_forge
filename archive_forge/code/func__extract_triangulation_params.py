import sys
import numpy as np
from matplotlib import _api
@staticmethod
def _extract_triangulation_params(args, kwargs):
    x, y, *args = args
    triangles = kwargs.pop('triangles', None)
    from_args = False
    if triangles is None and args:
        triangles = args[0]
        from_args = True
    if triangles is not None:
        try:
            triangles = np.asarray(triangles, dtype=np.int32)
        except ValueError:
            triangles = None
    if triangles is not None and (triangles.ndim != 2 or triangles.shape[1] != 3):
        triangles = None
    if triangles is not None and from_args:
        args = args[1:]
    mask = kwargs.pop('mask', None)
    return (x, y, triangles, mask, args, kwargs)