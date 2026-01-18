from __future__ import annotations
import functools
from typing import Optional
import torch
from . import _dtypes_impl, _util
from ._normalizations import (
Return a dtype that is real or complex floating-point.

    For inputs that are boolean or integer dtypes, this returns the default
    float dtype; inputs that are complex get converted to the default complex
    dtype; real floating-point dtypes (`float*`) get passed through unchanged
    