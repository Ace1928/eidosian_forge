from __future__ import annotations
import collections
import dataclasses
import datetime
import functools
import hashlib
import inspect
import io
import os
import pickle
import sys
import tempfile
import threading
import uuid
import weakref
from enum import Enum
from typing import Any, Callable, Dict, Final, Pattern, Type, Union
from typing_extensions import TypeAlias
from streamlit import type_util, util
from streamlit.errors import StreamlitAPIException
from streamlit.runtime.caching.cache_errors import UnhashableTypeError
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit.util import HASHLIB_KWARGS
def _float_to_bytes(f: float) -> bytes:
    import struct
    return struct.pack('<d', f)