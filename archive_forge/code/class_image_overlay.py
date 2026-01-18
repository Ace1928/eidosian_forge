import warnings
import numpy as np
import param
from packaging.version import Version
from param import _is_number
from ..core import (
from ..core.data import ArrayInterface, DictInterface, PandasInterface, default_datatype
from ..core.data.util import dask_array_module
from ..core.util import (
from ..element.chart import Histogram, Scatter
from ..element.path import Contours, Polygons
from ..element.raster import RGB, Image
from ..element.util import categorical_aggregate2d  # noqa (API import)
from ..streams import RangeXY
from ..util.locator import MaxNLocator
class image_overlay(Operation):
    """
    Operation to build a overlay of images to a specification from a
    subset of the required elements.

    This is useful for reordering the elements of an overlay,
    duplicating layers of an overlay or creating blank image elements
    in the appropriate positions.

    For instance, image_overlay may build a three layered input
    suitable for the RGB factory operation even if supplied with one
    or two of the required channels (creating blank channels for the
    missing elements).

    Note that if there is any ambiguity regarding the match, the
    strongest match will be used. In the case of a tie in match
    strength, the first layer in the input is used. One successful
    match is always required.
    """
    output_type = Overlay
    spec = param.String(doc='\n       Specification of the output Overlay structure. For instance:\n\n       Image.R * Image.G * Image.B\n\n       Will ensure an overlay of this structure is created even if\n       (for instance) only (Image.R * Image.B) is supplied.\n\n       Elements in the input overlay that match are placed in the\n       appropriate positions and unavailable specification elements\n       are created with the specified fill group.')
    fill = param.Number(default=0)
    default_range = param.Tuple(default=(0, 1), doc='\n        The default range that will be set on the value_dimension of\n        any automatically created blank image elements.')
    group = param.String(default='Transform', doc='\n        The group assigned to the resulting overlay.')

    @classmethod
    def _match(cls, el, spec):
        """Return the strength of the match (None if no match)"""
        spec_dict = dict(zip(['type', 'group', 'label'], spec.split('.')))
        if not isinstance(el, Image) or spec_dict['type'] != 'Image':
            raise NotImplementedError('Only Image currently supported')
        sanitizers = {'group': group_sanitizer, 'label': label_sanitizer}
        strength = 1
        for key in ['group', 'label']:
            attr_value = sanitizers[key](getattr(el, key))
            if key in spec_dict:
                if spec_dict[key] != attr_value:
                    return None
                strength += 1
        return strength

    def _match_overlay(self, raster, overlay_spec):
        """
        Given a raster or input overlay, generate a list of matched
        elements (None if no match) and corresponding tuple of match
        strength values.
        """
        ordering = [None] * len(overlay_spec)
        strengths = [0] * len(overlay_spec)
        elements = raster.values() if isinstance(raster, Overlay) else [raster]
        for el in elements:
            for pos in range(len(overlay_spec)):
                strength = self._match(el, overlay_spec[pos])
                if strength is None:
                    continue
                elif strength <= strengths[pos]:
                    continue
                else:
                    ordering[pos] = el
                    strengths[pos] = strength
        return (ordering, strengths)

    def _process(self, raster, key=None):
        specs = tuple((el.strip() for el in self.p.spec.split('*')))
        ordering, strengths = self._match_overlay(raster, specs)
        if all((el is None for el in ordering)):
            raise Exception('The image_overlay operation requires at least one match')
        completed = []
        strongest = ordering[np.argmax(strengths)]
        for el, spec in zip(ordering, specs):
            if el is None:
                spec_dict = dict(zip(['type', 'group', 'label'], spec.split('.')))
                el = Image(np.ones(strongest.data.shape) * self.p.fill, group=spec_dict.get('group', 'Image'), label=spec_dict.get('label', ''))
                el.vdims[0].range = self.p.default_range
            completed.append(el)
        return np.prod(completed)