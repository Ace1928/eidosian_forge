import logging
import os
from collections import defaultdict, namedtuple
from functools import reduce
from itertools import chain
from math import log2
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple
from fontTools.config import OPTIONS
from fontTools.misc.intTools import bit_count, bit_indices
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables import otBase, otTables
def _classDef_bytes(class_data: List[Tuple[List[Tuple[int, int]], int, int]], class_ids: List[int], coverage=False):
    if not class_ids:
        return 0
    first_ranges, min_glyph_id, max_glyph_id = class_data[class_ids[0]]
    range_count = len(first_ranges)
    for i in class_ids[1:]:
        data = class_data[i]
        range_count += len(data[0])
        min_glyph_id = min(min_glyph_id, data[1])
        max_glyph_id = max(max_glyph_id, data[2])
    glyphCount = max_glyph_id - min_glyph_id + 1
    format1_bytes = 6 + glyphCount * 2
    format2_bytes = 4 + range_count * 6
    return min(format1_bytes, format2_bytes)