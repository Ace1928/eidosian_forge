from __future__ import annotations
import hashlib
import io
import os
import re
from pathlib import Path
from streamlit import runtime
from streamlit.runtime import caching
def _handle_bytes_data(data: bytes) -> bytes:
    """Handles io.BytesIO data, converting SRT to VTT content if needed."""
    return _srt_to_vtt(data) if _is_srt(data) else data