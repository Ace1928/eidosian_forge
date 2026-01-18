from __future__ import annotations
import atexit
import builtins
import io
import logging
import math
import os
import re
import struct
import sys
import tempfile
import warnings
from collections.abc import Callable, MutableMapping
from enum import IntEnum
from pathlib import Path
from . import (
from ._binary import i32le, o32be, o32le
from ._util import DeferredError, is_path
def _repr_image(self, image_format, **kwargs):
    """Helper function for iPython display hook.

        :param image_format: Image format.
        :returns: image as bytes, saved into the given format.
        """
    b = io.BytesIO()
    try:
        self.save(b, image_format, **kwargs)
    except Exception:
        return None
    return b.getvalue()