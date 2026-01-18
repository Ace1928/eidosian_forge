from matplotlib import _api, _docstring
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxTransformTo
from matplotlib.transforms import IdentityTransform, TransformedBbox
from . import axes_size as Size
from .parasite_axes import HostAxes
class AnchoredZoomLocator(AnchoredLocatorBase):

    def __init__(self, parent_axes, zoom, loc, borderpad=0.5, bbox_to_anchor=None, bbox_transform=None):
        self.parent_axes = parent_axes
        self.zoom = zoom
        if bbox_to_anchor is None:
            bbox_to_anchor = parent_axes.bbox
        super().__init__(bbox_to_anchor, None, loc, borderpad=borderpad, bbox_transform=bbox_transform)

    def get_bbox(self, renderer):
        bb = self.parent_axes.transData.transform_bbox(self.axes.viewLim)
        fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
        pad = self.pad * fontsize
        return Bbox.from_bounds(0, 0, abs(bb.width * self.zoom), abs(bb.height * self.zoom)).padded(pad)