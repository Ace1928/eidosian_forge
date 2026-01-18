from __future__ import annotations
import collections
import re
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any
import numpy as np
import scipy.constants as const
def _check_mappings(u):
    for v in DERIVED_UNITS.values():
        for k2, v2 in v.items():
            if all((v2.get(ku, 0) == vu for ku, vu in u.items())) and all((u.get(kv2, 0) == vv2 for kv2, vv2 in v2.items())):
                return {k2: 1}
    return u