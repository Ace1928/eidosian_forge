import param
from holoviews.plotting.util import attach_streams
from ...core import AdjointLayout, Empty, GridMatrix, GridSpace, HoloMap, NdLayout
from ...core.options import Store
from ...core.util import wrap_tuple
from ...element import Histogram
from ..plot import (
from .util import configure_matching_axes_from_dims, figure_grid
def _trigger_refresh(self, key):
    """Triggers update to a plot on a refresh event"""
    if self.top_level:
        self.update(key)
    else:
        self.current_key = None
        self.current_frame = None