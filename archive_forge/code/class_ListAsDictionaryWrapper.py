from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.run import condition
from googlecloudsdk.core.console import console_attr
import six
class ListAsDictionaryWrapper(collections_abc.MutableMapping):
    """Wraps repeated messages field with name in a dict-like object.

  Operations in these classes are O(n) for simplicity. This needs to match the
  live state of the underlying list of messages, including edits made by others.
  """

    def __init__(self, to_wrap, key_field='name', filter_func=None):
        """Wraps list of messages to be accessible as a read-only dictionary.

    Arguments:
      to_wrap: List[Message], List of messages to treat as a dictionary.
      key_field: attribute to use as the keys of the dictionary
      filter_func: filter function to allow only considering certain messages
        from the wrapped list. This function should take a message as its only
        argument and return True if this message should be included.
    """
        self._m = to_wrap
        self._key_field = key_field
        self._filter = filter_func or (lambda _: True)

    def __getitem__(self, key):
        """Implements evaluation of `self[key]`."""
        for k, item in self.items():
            if k == key:
                return item
        raise KeyError(key)

    def __setitem__(self, key, value):
        setattr(value, self._key_field, key)
        for index, item in enumerate(self._m):
            if getattr(item, self._key_field) == key:
                if not self._filter(item):
                    raise KeyError(key)
                self._m[index] = value
                return
        self._m.append(value)

    def setdefault(self, key, default):
        for item in self._m:
            if getattr(item, self._key_field) == key:
                if not self._filter(item):
                    raise KeyError(key)
                return item
        setattr(default, self._key_field, key)
        self._m.append(default)
        return default

    def __delitem__(self, key):
        """Implements evaluation of `del self[key]`."""
        index_to_delete = None
        for index, item in enumerate(self._m):
            if getattr(item, self._key_field) == key:
                if self._filter(item):
                    index_to_delete = index
                break
        if index_to_delete is None:
            raise KeyError(key)
        del self._m[index_to_delete]

    def __len__(self):
        """Implements evaluation of `len(self)`."""
        return sum((1 for _ in self.items()))

    def __iter__(self):
        """Returns a generator yielding the message keys."""
        return (item[0] for item in self.items())

    def MakeSerializable(self):
        return self._m

    def __repr__(self):
        return '{}{{{}}}'.format(type(self).__name__, ', '.join(('{}: {}'.format(k, v) for k, v in self.items())))

    def items(self):
        return ListItemsView(self)

    def values(self):
        return ListValuesView(self)