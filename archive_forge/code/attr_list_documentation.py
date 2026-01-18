from __future__ import annotations
from typing import TYPE_CHECKING
from . import Extension
from ..treeprocessors import Treeprocessor
import re

        Sanitize name as 'an XML Name, minus the `:`.'
        See <https://www.w3.org/TR/REC-xml-names/#NT-NCName>.
        