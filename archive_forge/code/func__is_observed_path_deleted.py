from __future__ import annotations
import sys
from dataclasses import dataclass
from functools import reduce
import ctypes.wintypes  # noqa: E402
def _is_observed_path_deleted(handle, path):
    buff = ctypes.create_unicode_buffer(PATH_BUFFER_SIZE)
    GetFinalPathNameByHandleW(handle, buff, PATH_BUFFER_SIZE, VOLUME_NAME_NT)
    return buff.value != path