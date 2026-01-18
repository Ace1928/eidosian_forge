from __future__ import annotations
import hashlib
import io
import os
import re
from pathlib import Path
from streamlit import runtime
from streamlit.runtime import caching
def _srt_to_vtt(srt_data: str | bytes) -> bytes:
    """
    Convert subtitles from SubRip (.srt) format to WebVTT (.vtt) format.
    This function accepts the content of the .srt file either as a string
    or as a BytesIO stream.
    Parameters
    ----------
    srt_data : str or bytes
        The content of the .srt file as a string or a bytes stream.
    Returns
    -------
    bytes
        The content converted into .vtt format.
    """
    if isinstance(srt_data, bytes):
        try:
            srt_data = srt_data.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError('Could not decode the input stream as UTF-8.') from e
    if not isinstance(srt_data, str):
        raise TypeError(f'Input must be a string or a bytes stream, not {type(srt_data)}.')
    vtt_data = re.sub(SRT_CONVERSION_REGEX, '\\1.\\2', srt_data)
    vtt_content = 'WEBVTT\n\n' + vtt_data
    vtt_content = vtt_content.strip().encode('utf-8')
    return vtt_content