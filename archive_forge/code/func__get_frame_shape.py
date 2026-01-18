from fractions import Fraction
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union, Generator
import av
import av.filter
import numpy as np
from numpy.lib.stride_tricks import as_strided
from ..core import Request
from ..core.request import URI_BYTES, InitializationError, IOMode
from ..core.v3_plugin_api import ImageProperties, PluginV3
def _get_frame_shape(frame: av.VideoFrame) -> Tuple[int, ...]:
    """Compute the frame's array shape

    Parameters
    ----------
    frame : av.VideoFrame
        A frame for which the resulting shape should be computed.

    Returns
    -------
    shape : Tuple[int, ...]
        A tuple describing the shape of the image data in the frame.

    """
    widths = [component.width for component in frame.format.components]
    heights = [component.height for component in frame.format.components]
    bits = np.array([component.bits for component in frame.format.components])
    line_sizes = [plane.line_size for plane in frame.planes]
    subsampled_width = widths[:-1] != widths[1:]
    subsampled_height = heights[:-1] != heights[1:]
    unaligned_components = np.any(bits % 8 != 0) or line_sizes[:-1] != line_sizes[1:]
    if subsampled_width or subsampled_height or unaligned_components:
        raise IOError(f"{frame.format.name} can't be expressed as a strided array.Use `format=` to select a format to convert into.")
    shape = [frame.height, frame.width]
    n_planes = max([component.plane for component in frame.format.components]) + 1
    if n_planes > 1:
        shape = [n_planes] + shape
    channels_per_plane = [0] * n_planes
    for component in frame.format.components:
        channels_per_plane[component.plane] += 1
    n_channels = max(channels_per_plane)
    if n_channels > 1:
        shape = shape + [n_channels]
    return tuple(shape)