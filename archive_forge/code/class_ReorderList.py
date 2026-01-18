from fontTools import ttLib
from fontTools.ttLib.tables import otBase
from fontTools.ttLib.tables import otTables as ot
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import deque
from typing import (
@dataclass(frozen=True)
class ReorderList(ReorderRule):
    """Reorder the items within a list to match the updated glyph order.

    Useful when a list ordered by coverage itself contains something ordered by a gid.
    For example, the PairSet table of https://docs.microsoft.com/en-us/typography/opentype/spec/gpos#lookup-type-2-pair-adjustment-positioning-subtable.
    """
    list_attr: str
    key: str

    def apply(self, font: ttLib.TTFont, value: otBase.BaseTable) -> None:
        lst = _get_dotted_attr(value, self.list_attr)
        assert isinstance(lst, list), f'{self.list_attr} should be a list'
        lst.sort(key=lambda v: font.getGlyphID(getattr(v, self.key)))