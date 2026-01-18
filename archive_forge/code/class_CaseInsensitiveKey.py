import collections
from humanfriendly.compat import basestring, unicode
class CaseInsensitiveKey(unicode):
    """
    Simple case insensitive dictionary key implementation.

    The :class:`CaseInsensitiveKey` class provides an intentionally simple
    implementation of case insensitive strings to be used as dictionary keys.

    If you need features like Unicode normalization or proper case folding
    please consider using a more advanced implementation like the :pypi:`istr`
    package instead.
    """

    def __new__(cls, value):
        """Create a :class:`CaseInsensitiveKey` object."""
        obj = unicode.__new__(cls, value)
        normalized = obj.lower()
        obj._normalized = normalized
        obj._hash_value = hash(normalized)
        return obj

    def __hash__(self):
        """Get the hash value of the lowercased string."""
        return self._hash_value

    def __eq__(self, other):
        """Compare two strings as lowercase."""
        if isinstance(other, CaseInsensitiveKey):
            return self._normalized == other._normalized
        elif isinstance(other, unicode):
            return self._normalized == other.lower()
        else:
            return NotImplemented