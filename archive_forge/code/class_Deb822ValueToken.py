import re
import sys
from weakref import ReferenceType
import weakref
from debian._util import resolve_ref, _strI
from debian._deb822_repro._util import BufferingIterator
class Deb822ValueToken(Deb822Token):
    """A field value can be split into multi "Deb822ValueToken"s (as well as separator tokens)"""
    __slots__ = ()