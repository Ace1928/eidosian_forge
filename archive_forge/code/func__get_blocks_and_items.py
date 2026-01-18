from __future__ import annotations
from contextlib import suppress
import copy
from datetime import (
import itertools
import os
import re
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.computation.pytables import (
from pandas.core.construction import extract_array
from pandas.core.indexes.api import ensure_index
from pandas.core.internals import (
from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
@staticmethod
def _get_blocks_and_items(frame: DataFrame, table_exists: bool, new_non_index_axes, values_axes, data_columns):
    if isinstance(frame._mgr, ArrayManager):
        frame = frame._as_manager('block')

    def get_blk_items(mgr):
        return [mgr.items.take(blk.mgr_locs) for blk in mgr.blocks]
    mgr = frame._mgr
    mgr = cast(BlockManager, mgr)
    blocks: list[Block] = list(mgr.blocks)
    blk_items: list[Index] = get_blk_items(mgr)
    if len(data_columns):
        axis, axis_labels = new_non_index_axes[0]
        new_labels = Index(axis_labels).difference(Index(data_columns))
        mgr = frame.reindex(new_labels, axis=axis)._mgr
        mgr = cast(BlockManager, mgr)
        blocks = list(mgr.blocks)
        blk_items = get_blk_items(mgr)
        for c in data_columns:
            mgr = frame.reindex([c], axis=axis)._mgr
            mgr = cast(BlockManager, mgr)
            blocks.extend(mgr.blocks)
            blk_items.extend(get_blk_items(mgr))
    if table_exists:
        by_items = {tuple(b_items.tolist()): (b, b_items) for b, b_items in zip(blocks, blk_items)}
        new_blocks: list[Block] = []
        new_blk_items = []
        for ea in values_axes:
            items = tuple(ea.values)
            try:
                b, b_items = by_items.pop(items)
                new_blocks.append(b)
                new_blk_items.append(b_items)
            except (IndexError, KeyError) as err:
                jitems = ','.join([pprint_thing(item) for item in items])
                raise ValueError(f'cannot match existing table structure for [{jitems}] on appending data') from err
        blocks = new_blocks
        blk_items = new_blk_items
    return (blocks, blk_items)