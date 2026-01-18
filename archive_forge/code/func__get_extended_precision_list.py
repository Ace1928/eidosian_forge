from __future__ import annotations
from collections.abc import Iterable
from typing import Final, TYPE_CHECKING, Callable
import numpy as np
def _get_extended_precision_list() -> list[str]:
    extended_names = ['uint128', 'uint256', 'int128', 'int256', 'float80', 'float96', 'float128', 'float256', 'complex160', 'complex192', 'complex256', 'complex512']
    return [i for i in extended_names if hasattr(np, i)]