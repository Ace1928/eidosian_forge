import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
def _path_to_3d_segment_with_codes(path, zs=0, zdir='z'):
    """Convert a path to a 3D segment with path codes."""
    zs = np.broadcast_to(zs, len(path))
    pathsegs = path.iter_segments(simplify=False, curves=False)
    seg_codes = [((x, y, z), code) for ((x, y), code), z in zip(pathsegs, zs)]
    if seg_codes:
        seg, codes = zip(*seg_codes)
        seg3d = [juggle_axes(x, y, z, zdir) for x, y, z in seg]
    else:
        seg3d = []
        codes = []
    return (seg3d, list(codes))