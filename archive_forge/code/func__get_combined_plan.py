from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.missing import NA
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.internals.array_manager import ArrayManager
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import (
def _get_combined_plan(mgrs: list[BlockManager]) -> list[tuple[BlockPlacement, list[JoinUnit]]]:
    plan = []
    max_len = mgrs[0].shape[0]
    blknos_list = [mgr.blknos for mgr in mgrs]
    pairs = libinternals.get_concat_blkno_indexers(blknos_list)
    for ind, (blknos, bp) in enumerate(pairs):
        units_for_bp = []
        for k, mgr in enumerate(mgrs):
            blkno = blknos[k]
            nb = _get_block_for_concat_plan(mgr, bp, blkno, max_len=max_len)
            unit = JoinUnit(nb)
            units_for_bp.append(unit)
        plan.append((bp, units_for_bp))
    return plan