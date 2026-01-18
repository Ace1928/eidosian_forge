from __future__ import annotations
import hashlib
import io
import os
import re
from pathlib import Path
from streamlit import runtime
from streamlit.runtime import caching
def _handle_string_or_path_data(data_or_path: str | Path) -> bytes:
    """Handles string data, either as a file path or raw content."""
    if os.path.isfile(data_or_path):
        path = Path(data_or_path)
        file_extension = path.suffix.lower()
        if file_extension not in SUBTITLE_ALLOWED_FORMATS:
            raise ValueError(f'Incorrect subtitle format {file_extension}. Subtitles must be in one of the following formats: {', '.join(SUBTITLE_ALLOWED_FORMATS)}')
        with open(data_or_path, 'rb') as file:
            content = file.read()
        return _srt_to_vtt(content) if file_extension == '.srt' else content
    elif isinstance(data_or_path, Path):
        raise ValueError(f'File {data_or_path} does not exist.')
    content_string = data_or_path.strip()
    if content_string.startswith('WEBVTT') or content_string == '':
        return content_string.encode('utf-8')
    elif _is_srt(content_string):
        return _srt_to_vtt(content_string)
    raise ValueError('The provided string neither matches valid VTT nor SRT format.')