from collections import Counter
import os
import re
import sys
import traceback
import warnings
from .builder import (
from .dammit import UnicodeDammit
from .element import (
def _linkage_fixer(self, el):
    """Make sure linkage of this fragment is sound."""
    first = el.contents[0]
    child = el.contents[-1]
    descendant = child
    if child is first and el.parent is not None:
        el.next_element = child
        prev_el = child.previous_element
        if prev_el is not None and prev_el is not el:
            prev_el.next_element = None
        child.previous_element = el
        child.previous_sibling = None
    child.next_sibling = None
    if isinstance(child, Tag) and child.contents:
        descendant = child._last_descendant(False)
    descendant.next_element = None
    descendant.next_sibling = None
    target = el
    while True:
        if target is None:
            break
        elif target.next_sibling is not None:
            descendant.next_element = target.next_sibling
            target.next_sibling.previous_element = child
            break
        target = target.parent