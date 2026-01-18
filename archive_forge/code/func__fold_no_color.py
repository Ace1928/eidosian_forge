from __future__ import annotations
import base64
import os
import platform
import sys
from functools import reduce
from typing import Any
def _fold_no_color(self, a: Any, b: Any) -> str:
    try:
        A = a.no_color()
    except AttributeError:
        A = str(a)
    try:
        B = b.no_color()
    except AttributeError:
        B = str(b)
    return f'{A}{B}'