from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _get_dividers(self, options):
    """Return only those dividers that should be printed, based on slicing.

        Arguments:

        options - dictionary of option settings."""
    import copy
    if options['oldsortslice']:
        dividers = copy.deepcopy(self._dividers[options['start']:options['end']])
    else:
        dividers = copy.deepcopy(self._dividers)
    if options['sortby']:
        dividers = [False for divider in dividers]
    return dividers