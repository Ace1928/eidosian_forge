import param
import numpy as np
from holoviews.plotting.mpl.element import ColorbarPlot
from holoviews.util.transform import dim
from holoviews.core.options import abbreviated_exception
def _get_us_vs(self, element):
    radians = element.dimension_values(2) if len(element.data) else []
    mag_dim = element.get_dimension(3)
    if isinstance(mag_dim, dim):
        magnitudes = mag_dim.apply(element, flat=True)
    else:
        magnitudes = element.dimension_values(mag_dim)
    if self.convention == 'to':
        radians -= np.pi
    if self.invert_axes:
        radians -= 0.5 * np.pi
    us = -magnitudes * np.sin(radians.flatten())
    vs = -magnitudes * np.cos(radians.flatten())
    return (us, vs)