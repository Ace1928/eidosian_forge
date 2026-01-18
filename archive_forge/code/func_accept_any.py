from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
def accept_any(self, tids: T.Tuple[str, ...]) -> str:
    tid = self.current.tid
    if tid in tids:
        self.getsym()
        return tid
    return ''