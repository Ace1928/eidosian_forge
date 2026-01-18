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
def _parse_start_time_end_time(start_time: MediaTime, end_time: MediaTime | None) -> tuple[int, int | None]:
    """Parse start_time and end_time and return them as int."""
    try:
        maybe_start_time = duration_to_seconds(start_time, coerce_none_to_inf=False)
        if maybe_start_time is None:
            raise ValueError
        start_time = int(maybe_start_time)
    except (StreamlitAPIException, ValueError):
        error_msg = TIMEDELTA_PARSE_ERROR_MESSAGE.format(param_name='start_time', param_value=start_time)
        raise StreamlitAPIException(error_msg) from None
    try:
        end_time = duration_to_seconds(end_time, coerce_none_to_inf=False)
        if end_time is not None:
            end_time = int(end_time)
    except StreamlitAPIException:
        error_msg = TIMEDELTA_PARSE_ERROR_MESSAGE.format(param_name='end_time', param_value=end_time)
        raise StreamlitAPIException(error_msg) from None
    return (start_time, end_time)