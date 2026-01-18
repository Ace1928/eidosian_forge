from __future__ import annotations
import base64
import os
import platform
import sys
from functools import reduce
from typing import Any
def _read_as_base64(path: str) -> str:
    with open(path, mode='rb') as fh:
        encoded = base64.b64encode(fh.read())
        return encoded.decode('ascii')