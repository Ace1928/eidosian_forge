from __future__ import annotations
import hashlib
import os
import time
from pathlib import Path
from streamlit.util import HASHLIB_KWARGS
def _get_file_content_with_blocking_retries(file_path: str) -> bytes:
    content = b''
    for i in range(_MAX_RETRIES):
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                break
        except FileNotFoundError as e:
            if i >= _MAX_RETRIES - 1:
                raise e
            time.sleep(_RETRY_WAIT_SECS)
    return content