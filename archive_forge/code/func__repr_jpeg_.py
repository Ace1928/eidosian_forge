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
def _repr_jpeg_(self):
    """iPython display hook support for JPEG format.

        :returns: JPEG version of the image as bytes
        """
    return self._repr_image('JPEG')