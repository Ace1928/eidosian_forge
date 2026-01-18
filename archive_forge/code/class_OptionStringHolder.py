from __future__ import annotations
import re
import os
import typing as T
from ...mesonlib import version_compare
from ...interpreterbase import (
class OptionStringHolder(StringHolder):
    held_object: OptionString

    def op_div(self, other: str) -> T.Union[str, OptionString]:
        ret = super().op_div(other)
        name = self._op_div(self.held_object.optname, other)
        return OptionString(ret, name)