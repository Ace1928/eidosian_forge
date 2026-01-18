import re
import sys
from weakref import ReferenceType
import weakref
from debian._util import resolve_ref, _strI
from debian._deb822_repro._util import BufferingIterator
class Deb822FieldNameToken(Deb822Token):
    __slots__ = ()

    def __init__(self, text):
        if not isinstance(text, _strI):
            text = _strI(sys.intern(text))
        super().__init__(text)

    @property
    def text(self):
        return cast('_strI', self._text)