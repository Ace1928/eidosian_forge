import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
class _DictOfContainers(collections.defaultdict):
    """A defaultdict with customized equality checks that ignore empty values.

    Non-empty value is checked by: `bool(value_item) == True`.
    """

    def __eq__(self, other):
        if isinstance(other, _DictOfContainers):
            mine = self._non_empty_items()
            theirs = other._non_empty_items()
            return mine == theirs
        return NotImplemented

    def __ne__(self, other):
        ret = self.__eq__(other)
        if ret is NotImplemented:
            return ret
        else:
            return not ret

    def _non_empty_items(self):
        return [(k, vs) for k, vs in sorted(self.items()) if vs]