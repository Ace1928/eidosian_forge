import copy
from collections.abc import Sized
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import IdentityTransform, Affine2D
from ._enums import JoinStyle, CapStyle
def _set_marker(self, marker):
    """
        Set the marker.

        Parameters
        ----------
        marker : str, array-like, Path, MarkerStyle, or None, default: None
            - Another instance of *MarkerStyle* copies the details of that
              ``marker``.
            - *None* means no marker.
            - For other possible marker values see the module docstring
              `matplotlib.markers`.
        """
    if isinstance(marker, str) and cbook.is_math_text(marker):
        self._marker_function = self._set_mathtext_path
    elif isinstance(marker, (int, str)) and marker in self.markers:
        self._marker_function = getattr(self, '_set_' + self.markers[marker])
    elif isinstance(marker, np.ndarray) and marker.ndim == 2 and (marker.shape[1] == 2):
        self._marker_function = self._set_vertices
    elif isinstance(marker, Path):
        self._marker_function = self._set_path_marker
    elif isinstance(marker, Sized) and len(marker) in (2, 3) and (marker[1] in (0, 1, 2)):
        self._marker_function = self._set_tuple_marker
    elif isinstance(marker, MarkerStyle):
        self.__dict__ = copy.deepcopy(marker.__dict__)
    else:
        try:
            Path(marker)
            self._marker_function = self._set_vertices
        except ValueError as err:
            raise ValueError(f'Unrecognized marker style {marker!r}') from err
    if not isinstance(marker, MarkerStyle):
        self._marker = marker
        self._recache()