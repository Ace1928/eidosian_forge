from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Optional
from pyglet import event
from pyglet.text import runlist
def _delete_text(self, start, end):
    super()._delete_text(start, end)
    for runs in self._style_runs.values():
        runs.delete(start, end)