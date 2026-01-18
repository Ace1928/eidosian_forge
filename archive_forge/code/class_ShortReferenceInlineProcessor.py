from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
class ShortReferenceInlineProcessor(ReferenceInlineProcessor):
    """Short form of reference: `[google]`. """

    def evalId(self, data: str, index: int, text: str) -> tuple[str, int, bool]:
        """Evaluate the id of `[ref]`.  """
        return (text.lower(), index, True)