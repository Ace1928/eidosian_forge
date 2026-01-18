from __future__ import annotations
import io
import re
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Final, Union, cast
from typing_extensions import TypeAlias
import streamlit as st
from streamlit import runtime, type_util, url_util
from streamlit.elements.lib.subtitle_utils import process_subtitle_data
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Audio_pb2 import Audio as AudioProto
from streamlit.proto.Video_pb2 import Video as VideoProto
from streamlit.runtime import caching
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.runtime_util import duration_to_seconds
def _validate_and_normalize(data: npt.NDArray[Any]) -> tuple[bytes, int]:
    """Validates and normalizes numpy array data.
    We validate numpy array shape (should be 1d or 2d)
    We normalize input data to int16 [-32768, 32767] range.

    Parameters
    ----------
    data : numpy array
        numpy array to be validated and normalized

    Returns
    -------
    Tuple of (bytes, int)
        (bytes, nchan)
        where
         - bytes : bytes of normalized numpy array converted to int16
         - nchan : number of channels for audio signal. 1 for mono, or 2 for stereo.
    """
    import numpy as np
    data: npt.NDArray[Any] = np.array(data, dtype=float)
    if len(data.shape) == 1:
        nchan = 1
    elif len(data.shape) == 2:
        nchan = data.shape[0]
        data = data.T.ravel()
    else:
        raise StreamlitAPIException('Numpy array audio input must be a 1D or 2D array.')
    if data.size == 0:
        return (data.astype(np.int16).tobytes(), nchan)
    max_abs_value = np.max(np.abs(data))
    np_array = data / max_abs_value * 32767
    scaled_data = np_array.astype(np.int16)
    return (scaled_data.tobytes(), nchan)