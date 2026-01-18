from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
class EmStrongItem(NamedTuple):
    """Emphasis/strong pattern item."""
    pattern: re.Pattern[str]
    builder: str
    tags: str