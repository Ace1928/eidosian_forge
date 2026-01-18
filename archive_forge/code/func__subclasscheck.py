from __future__ import annotations
from typing import (
@classmethod
def _subclasscheck(cls, inst) -> bool:
    if not isinstance(inst, type):
        raise TypeError('issubclass() arg 1 must be a class')
    return _check(inst)