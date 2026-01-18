from fontTools.config import OPTIONS
from fontTools.misc.textTools import Tag, bytesjoin
from .DefaultTable import DefaultTable
from enum import IntEnum
import sys
import array
import struct
import logging
from functools import lru_cache
from typing import Iterator, NamedTuple, Optional, Tuple
def _gatherGraphForHarfbuzz(self, tables, obj_list, done, objidx, virtual_edges):
    real_links = []
    virtual_links = []
    item_idx = objidx
    for idx in virtual_edges:
        virtual_links.append((0, 0, idx))
    sortCoverageLast = False
    coverage_idx = 0
    if hasattr(self, 'sortCoverageLast'):
        for i, item in enumerate(self.items):
            if getattr(item, 'name', None) == 'Coverage':
                sortCoverageLast = True
                if id(item) not in done:
                    coverage_idx = item_idx = item._gatherGraphForHarfbuzz(tables, obj_list, done, item_idx, virtual_edges)
                else:
                    coverage_idx = done[id(item)]
                virtual_edges.append(coverage_idx)
                break
    child_idx = 0
    offset_pos = 0
    for i, item in enumerate(self.items):
        if hasattr(item, 'subWriter'):
            pos = offset_pos
        elif hasattr(item, 'getCountData'):
            offset_pos += item.size
            continue
        else:
            offset_pos = offset_pos + len(item)
            continue
        if id(item.subWriter) not in done:
            child_idx = item_idx = item.subWriter._gatherGraphForHarfbuzz(tables, obj_list, done, item_idx, virtual_edges)
        else:
            child_idx = done[id(item.subWriter)]
        real_edge = (pos, item.offsetSize, child_idx)
        real_links.append(real_edge)
        offset_pos += item.offsetSize
    tables.append(self)
    obj_list.append((real_links, virtual_links))
    item_idx += 1
    done[id(self)] = item_idx
    if sortCoverageLast:
        virtual_edges.pop()
    return item_idx