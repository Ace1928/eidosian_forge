import re
import sys
from weakref import ReferenceType
import weakref
from debian._util import resolve_ref, _strI
from debian._deb822_repro._util import BufferingIterator
class Deb822ValueContinuationToken(Deb822SemanticallySignificantWhiteSpace):
    """The whitespace denoting a value spanning an additional line (the first space on a line)"""
    __slots__ = ()