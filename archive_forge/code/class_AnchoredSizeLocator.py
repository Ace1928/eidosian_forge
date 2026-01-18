from matplotlib import _api, _docstring
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxTransformTo
from matplotlib.transforms import IdentityTransform, TransformedBbox
from . import axes_size as Size
from .parasite_axes import HostAxes
class AnchoredSizeLocator(AnchoredLocatorBase):

    def __init__(self, bbox_to_anchor, x_size, y_size, loc, borderpad=0.5, bbox_transform=None):
        super().__init__(bbox_to_anchor, None, loc, borderpad=borderpad, bbox_transform=bbox_transform)
        self.x_size = Size.from_any(x_size)
        self.y_size = Size.from_any(y_size)

    def get_bbox(self, renderer):
        bbox = self.get_bbox_to_anchor()
        dpi = renderer.points_to_pixels(72.0)
        r, a = self.x_size.get_size(renderer)
        width = bbox.width * r + a * dpi
        r, a = self.y_size.get_size(renderer)
        height = bbox.height * r + a * dpi
        fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
        pad = self.pad * fontsize
        return Bbox.from_bounds(0, 0, width, height).padded(pad)