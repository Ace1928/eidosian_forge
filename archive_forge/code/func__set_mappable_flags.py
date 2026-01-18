import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def _set_mappable_flags(self):
    """
        Determine whether edges and/or faces are color-mapped.

        This is a helper for update_scalarmappable.
        It sets Boolean flags '_edge_is_mapped' and '_face_is_mapped'.

        Returns
        -------
        mapping_change : bool
            True if either flag is True, or if a flag has changed.
        """
    edge0 = self._edge_is_mapped
    face0 = self._face_is_mapped
    self._edge_is_mapped = False
    self._face_is_mapped = False
    if self._A is not None:
        if not cbook._str_equal(self._original_facecolor, 'none'):
            self._face_is_mapped = True
            if cbook._str_equal(self._original_edgecolor, 'face'):
                self._edge_is_mapped = True
        elif self._original_edgecolor is None:
            self._edge_is_mapped = True
    mapped = self._face_is_mapped or self._edge_is_mapped
    changed = edge0 is None or face0 is None or self._edge_is_mapped != edge0 or (self._face_is_mapped != face0)
    return mapped or changed