import warnings
from warnings import warn
import breezy
class _DeprecatedList(list):
    __doc__ = list.__doc__ + msg
    is_deprecated = True

    def _warn_deprecated(self, func, *args, **kwargs):
        warn(msg, DeprecationWarning, stacklevel=3)
        return func(self, *args, **kwargs)

    def append(self, obj):
        'appending to {} is deprecated'.format(variable_name)
        return self._warn_deprecated(list.append, obj)

    def insert(self, index, obj):
        'inserting to {} is deprecated'.format(variable_name)
        return self._warn_deprecated(list.insert, index, obj)

    def extend(self, iterable):
        'extending {} is deprecated'.format(variable_name)
        return self._warn_deprecated(list.extend, iterable)

    def remove(self, value):
        'removing from {} is deprecated'.format(variable_name)
        return self._warn_deprecated(list.remove, value)

    def pop(self, index=None):
        "pop'ing from {} is deprecated".format(variable_name)
        if index:
            return self._warn_deprecated(list.pop, index)
        else:
            return self._warn_deprecated(list.pop)