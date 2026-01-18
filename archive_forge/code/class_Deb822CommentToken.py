import re
import sys
from weakref import ReferenceType
import weakref
from debian._util import resolve_ref, _strI
from debian._deb822_repro._util import BufferingIterator
class Deb822CommentToken(Deb822Token):
    __slots__ = ()

    @property
    def is_comment(self):
        return True