from matplotlib import _api, _docstring
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxTransformTo
from matplotlib.transforms import IdentityTransform, TransformedBbox
from . import axes_size as Size
from .parasite_axes import HostAxes
class BboxPatch(Patch):

    @_docstring.dedent_interpd
    def __init__(self, bbox, **kwargs):
        """
        Patch showing the shape bounded by a Bbox.

        Parameters
        ----------
        bbox : `~matplotlib.transforms.Bbox`
            Bbox to use for the extents of this patch.

        **kwargs
            Patch properties. Valid arguments include:

            %(Patch:kwdoc)s
        """
        if 'transform' in kwargs:
            raise ValueError('transform should not be set')
        kwargs['transform'] = IdentityTransform()
        super().__init__(**kwargs)
        self.bbox = bbox

    def get_path(self):
        x0, y0, x1, y1 = self.bbox.extents
        return Path._create_closed([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])