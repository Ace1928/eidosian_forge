from __future__ import annotations
import hashlib
import io
import os
import re
from pathlib import Path
from streamlit import runtime
from streamlit.runtime import caching
def _is_srt(stream: str | io.BytesIO | bytes) -> bool:
    if isinstance(stream, bytes):
        stream = io.BytesIO(stream)
    if isinstance(stream, str):
        stream = io.BytesIO(stream.encode('utf-8'))
    stream.seek(0)
    header = stream.read(33)
    try:
        header_str = header.decode('utf-8').strip()
    except UnicodeDecodeError:
        return False
    lines = header_str.split('\n')
    if len(lines) >= 2 and lines[0].isdigit():
        match = re.search(SRT_VALIDATION_REGEX, lines[1])
        if match:
            return True
    return False