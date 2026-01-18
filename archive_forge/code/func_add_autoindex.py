from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def add_autoindex(self, fieldname: str='Index') -> None:
    """Add an auto-incrementing index column to the table.
        Arguments:
        fieldname - name of the field to contain the new column of data"""
    self._field_names.insert(0, fieldname)
    self._align[fieldname] = self.align
    self._valign[fieldname] = self.valign
    for i, row in enumerate(self._rows):
        row.insert(0, i + 1)