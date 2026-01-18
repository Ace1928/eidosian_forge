from __future__ import annotations
import contextlib
from abc import ABC
from typing import (
def _default_factory() -> KDBIndexT:
    return cls(builder=builder, **kwargs)