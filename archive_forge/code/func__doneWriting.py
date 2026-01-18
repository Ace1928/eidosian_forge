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
def _doneWriting(self, internedTables, shareExtension=False):
    isExtension = hasattr(self, 'Extension')
    dontShare = hasattr(self, 'DontShare')
    if isExtension and (not shareExtension):
        internedTables = {}
    items = self.items
    for i in range(len(items)):
        item = items[i]
        if hasattr(item, 'getCountData'):
            items[i] = item.getCountData()
        elif hasattr(item, 'subWriter'):
            item.subWriter._doneWriting(internedTables, shareExtension=shareExtension)
            if not dontShare:
                items[i].subWriter = internedTables.setdefault(item.subWriter, item.subWriter)
    self.items = tuple(items)