from collections import defaultdict
import itertools
import re
import warnings
import sys
from bs4.element import (
from . import _htmlparser
def _root_tag_encountered(self, name):
    """Call this when you encounter the document's root tag.

        This is where we actually check whether an XML document is
        being incorrectly parsed as HTML, and issue the warning.
        """
    if self._root_tag is not None:
        return
    self._root_tag = name
    if name != 'html' and self._first_processing_instruction is not None and self._first_processing_instruction.lower().startswith('xml '):
        self._warn()