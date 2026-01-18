from __future__ import annotations
from . import Extension
from ..blockprocessors import OListProcessor, UListProcessor
import re
from typing import TYPE_CHECKING
class SaneOListProcessor(OListProcessor):
    """ Override `SIBLING_TAGS` to not include `ul` and set `LAZY_OL` to `False`. """
    SIBLING_TAGS = ['ol']
    ' Exclude `ul` from list of siblings. '
    LAZY_OL = False
    ' Disable lazy list behavior. '

    def __init__(self, parser: blockparser.BlockParser):
        super().__init__(parser)
        self.CHILD_RE = re.compile('^[ ]{0,%d}((\\d+\\.))[ ]+(.*)' % (self.tab_length - 1))