from __future__ import annotations
from typing import Any
import operator
from sympy.polys.polyutils import PicklableWithSlots
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains.domainelement import DomainElement
from sympy.utilities import public
@classmethod
def _get_val(cls, other):
    if isinstance(other, cls):
        return other.val
    else:
        try:
            return cls.dom.convert(other)
        except CoercionFailed:
            return None