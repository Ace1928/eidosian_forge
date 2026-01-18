import copy
from collections.abc import Sized
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import IdentityTransform, Affine2D
from ._enums import JoinStyle, CapStyle
def _set_mathtext_path(self):
    """
        Draw mathtext markers '$...$' using `.TextPath` object.

        Submitted by tcb
        """
    from matplotlib.text import TextPath
    text = TextPath(xy=(0, 0), s=self.get_marker(), usetex=mpl.rcParams['text.usetex'])
    if len(text.vertices) == 0:
        return
    bbox = text.get_extents()
    max_dim = max(bbox.width, bbox.height)
    self._transform = Affine2D().translate(-bbox.xmin + 0.5 * -bbox.width, -bbox.ymin + 0.5 * -bbox.height).scale(1.0 / max_dim)
    self._path = text
    self._snap = False