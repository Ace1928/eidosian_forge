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
def _flush_writer(self):
    """Flush the filter and encoder

        This will reset the filter to `None` and send EoF to the encoder,
        i.e., after calling, no more frames may be written.

        """
    stream = self._video_stream
    if self._video_filter is not None:
        for av_frame in self._video_filter:
            if stream.frames == 0:
                stream.width = av_frame.width
                stream.height = av_frame.height
            for packet in stream.encode(av_frame):
                self._container.mux(packet)
        self._video_filter = None
    for packet in stream.encode():
        self._container.mux(packet)
    self._video_stream = None