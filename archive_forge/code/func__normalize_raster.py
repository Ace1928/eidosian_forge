import param
from ..core import Overlay
from ..core.operation import Operation
from ..core.util import match_spec
from ..element import Raster
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