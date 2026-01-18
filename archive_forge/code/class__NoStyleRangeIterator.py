from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Optional
from pyglet import event
from pyglet.text import runlist
class _NoStyleRangeIterator:

    @staticmethod
    def ranges(start, end):
        yield (start, end, None)

    def __getitem__(self, index):
        return None