from functools import total_ordering
from ._funcs import astuple
from ._make import attrib, attrs
def _ensure_tuple(self, other):
    """
        Ensure *other* is a tuple of a valid length.

        Returns a possibly transformed *other* and ourselves as a tuple of
        the same length as *other*.
        """
    if self.__class__ is other.__class__:
        other = astuple(other)
    if not isinstance(other, tuple):
        raise NotImplementedError
    if not 1 <= len(other) <= 4:
        raise NotImplementedError
    return (astuple(self)[:len(other)], other)