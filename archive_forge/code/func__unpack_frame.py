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
def _unpack_frame(self, frame: av.VideoFrame, *, format: str=None) -> np.ndarray:
    """Convert a av.VideoFrame into a ndarray

        Parameters
        ----------
        frame : av.VideoFrame
            The frame to unpack.
        format : str
            If not None, convert the frame to the given format before unpacking.

        """
    if format is not None:
        frame = frame.reformat(format=format)
    dtype = _format_to_dtype(frame.format)
    shape = _get_frame_shape(frame)
    planes = list()
    for idx in range(len(frame.planes)):
        n_channels = sum([x.bits // (dtype.itemsize * 8) for x in frame.format.components if x.plane == idx])
        av_plane = frame.planes[idx]
        plane_shape = (av_plane.height, av_plane.width)
        plane_strides = (av_plane.line_size, n_channels * dtype.itemsize)
        if n_channels > 1:
            plane_shape += (n_channels,)
            plane_strides += (dtype.itemsize,)
        np_plane = as_strided(np.frombuffer(av_plane, dtype=dtype), shape=plane_shape, strides=plane_strides)
        planes.append(np_plane)
    if len(planes) > 1:
        out = np.concatenate(planes).reshape(shape)
    else:
        out = planes[0]
    return out