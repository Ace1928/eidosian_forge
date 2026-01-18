from __future__ import annotations
import re
import os
import typing as T
from ...mesonlib import version_compare
from ...interpreterbase import (
class OptionString(str):
    optname: str

    def __new__(cls, value: str, name: str) -> 'OptionString':
        obj = str.__new__(cls, value)
        obj.optname = name
        return obj

    def __getnewargs__(self) -> T.Tuple[str, str]:
        return (str(self), self.optname)