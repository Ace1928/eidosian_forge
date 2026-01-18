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
def _make_wav(data: npt.NDArray[Any], sample_rate: int) -> bytes:
    """
    Transform a numpy array to a PCM bytestring
    We use code from IPython display module to convert numpy array to wave bytes
    https://github.com/ipython/ipython/blob/1015c392f3d50cf4ff3e9f29beede8c1abfdcb2a/IPython/lib/display.py#L146
    """
    import wave
    scaled, nchan = _validate_and_normalize(data)
    with io.BytesIO() as fp, wave.open(fp, mode='wb') as waveobj:
        waveobj.setnchannels(nchan)
        waveobj.setframerate(sample_rate)
        waveobj.setsampwidth(2)
        waveobj.setcomptype('NONE', 'NONE')
        waveobj.writeframes(scaled)
        return fp.getvalue()