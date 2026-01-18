import collections
import collections.abc
import operator
import warnings
def __check_version__(self):
    """Raise InvalidOperationException if version is greater than 1 or policy contains conditions."""
    raise_version = self.version is not None and self.version > 1
    if raise_version or self._contains_conditions():
        raise InvalidOperationException(_DICT_ACCESS_MSG)