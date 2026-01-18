from __future__ import annotations
from typing import TYPE_CHECKING
from . import Extension
from ..treeprocessors import Treeprocessor
import re
def _handle_single_quote(s, t):
    k, v = t.split('=', 1)
    return (k, v.strip("'"))