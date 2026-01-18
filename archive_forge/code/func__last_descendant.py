import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
def _last_descendant(self, is_initialized=True, accept_self=True):
    """Finds the last element beneath this object to be parsed.

        :param is_initialized: Has `setup` been called on this PageElement
            yet?
        :param accept_self: Is `self` an acceptable answer to the question?
        """
    if is_initialized and self.next_sibling is not None:
        last_child = self.next_sibling.previous_element
    else:
        last_child = self
        while isinstance(last_child, Tag) and last_child.contents:
            last_child = last_child.contents[-1]
    if not accept_self and last_child is self:
        last_child = None
    return last_child