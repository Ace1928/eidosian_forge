import re
import sys
from weakref import ReferenceType
import weakref
from debian._util import resolve_ref, _strI
from debian._deb822_repro._util import BufferingIterator
class Deb822SemanticallySignificantWhiteSpace(Deb822WhitespaceToken):
    """Whitespace that (if removed) would change the meaning of the file (or cause syntax errors)"""
    __slots__ = ()