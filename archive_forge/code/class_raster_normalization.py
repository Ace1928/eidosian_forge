import param
from ..core import Overlay
from ..core.operation import Operation
from ..core.util import match_spec
from ..element import Raster
class raster_normalization(Normalization):
    """
    Normalizes elements of type Raster.

    For Raster elements containing (NxM) data, this will normalize the
    array/matrix into the specified range if value_dimension matches
    a key in the ranges dictionary.

    For elements containing (NxMxD) data, the (NxM) components of the
    third dimensional are normalized independently if the
    corresponding value dimensions are selected by the ranges
    dictionary.
    """

    def _process(self, raster, key=None):
        if isinstance(raster, Raster):
            return self._normalize_raster(raster, key)
        elif isinstance(raster, Overlay):
            overlay_clone = raster.clone(shared_data=False)
            for k, el in raster.items():
                overlay_clone[k] = self._normalize_raster(el, key)
            return overlay_clone
        else:
            raise ValueError('Input element must be a Raster or subclass of Raster.')

    def _normalize_raster(self, raster, key):
        if not isinstance(raster, Raster):
            return raster
        norm_raster = raster.clone(raster.data.copy())
        ranges = self.get_ranges(raster, key)
        for depth, name in enumerate((d.name for d in raster.vdims)):
            depth_range = ranges.get(name, (None, None))
            if None in depth_range:
                continue
            if depth_range and len(norm_raster.data.shape) == 2:
                depth_range = ranges[name]
                norm_raster.data[:, :] -= depth_range[0]
                range = depth_range[1] - depth_range[0]
                if range:
                    norm_raster.data[:, :] /= range
            elif depth_range:
                norm_raster.data[:, :, depth] -= depth_range[0]
                range = depth_range[1] - depth_range[0]
                if range:
                    norm_raster.data[:, :, depth] /= range
        return norm_raster