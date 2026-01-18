from __future__ import annotations
import typing
from collections import OrderedDict
from enum import Enum, auto
from threading import RLock
def _has_value_for_header(self, header_name: str, potential_value: str) -> bool:
    if header_name in self:
        return potential_value in self._container[header_name.lower()][1:]
    return False