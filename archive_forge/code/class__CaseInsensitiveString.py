import weakref
from weakref import ReferenceType
class _CaseInsensitiveString(str):
    """Case insensitive string.
    """
    __slots__ = ['str_lower']
    if TYPE_CHECKING:

        def __init__(self, s):
            super(_CaseInsensitiveString, self).__init__(s)
            self.str_lower = ''

    def __new__(cls, str_):
        s = str.__new__(cls, str_)
        s.str_lower = str_.lower()
        return s

    def __hash__(self):
        return hash(self.str_lower)

    def __eq__(self, other):
        try:
            return self.str_lower == other.lower()
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    def lower(self):
        return self.str_lower