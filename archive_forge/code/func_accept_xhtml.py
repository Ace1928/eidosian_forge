from __future__ import annotations
import codecs
import re
from .structures import ImmutableList
@property
def accept_xhtml(self):
    """True if this object accepts XHTML."""
    return 'application/xhtml+xml' in self or 'application/xml' in self