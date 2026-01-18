from fontTools import ttLib
from fontTools.ttLib.tables import otBase
from fontTools.ttLib.tables import otTables as ot
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import deque
from typing import (
def _sort_by_gid(get_glyph_id: Callable[[str], int], glyphs: List[str], parallel_list: Optional[List[Any]]):
    if parallel_list:
        reordered = sorted(((g, e) for g, e in zip(glyphs, parallel_list)), key=lambda t: get_glyph_id(t[0]))
        sorted_glyphs, sorted_parallel_list = map(list, zip(*reordered))
        parallel_list[:] = sorted_parallel_list
    else:
        sorted_glyphs = sorted(glyphs, key=get_glyph_id)
    glyphs[:] = sorted_glyphs