from __future__ import annotations
import codecs
import re
from .structures import ImmutableList
@property
def accept_json(self):
    """True if this object accepts JSON."""
    return 'application/json' in self