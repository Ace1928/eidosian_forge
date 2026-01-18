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
def _gatherTables(self, tables, extTables, done):
    done[id(self)] = True
    numItems = len(self.items)
    iRange = list(range(numItems))
    iRange.reverse()
    isExtension = hasattr(self, 'Extension')
    selfTables = tables
    if isExtension:
        assert extTables is not None, 'Program or XML editing error. Extension subtables cannot contain extensions subtables'
        tables, extTables, done = (extTables, None, {})
    sortCoverageLast = False
    if hasattr(self, 'sortCoverageLast'):
        for i in range(numItems):
            item = self.items[i]
            if hasattr(item, 'subWriter') and getattr(item.subWriter, 'name', None) == 'Coverage':
                sortCoverageLast = True
                break
        if id(item.subWriter) not in done:
            item.subWriter._gatherTables(tables, extTables, done)
        else:
            pass
    for i in iRange:
        item = self.items[i]
        if not hasattr(item, 'subWriter'):
            continue
        if sortCoverageLast and i == 1 and (getattr(item.subWriter, 'name', None) == 'Coverage'):
            continue
        if id(item.subWriter) not in done:
            item.subWriter._gatherTables(tables, extTables, done)
        else:
            pass
    selfTables.append(self)