import collections
import copy
import sys
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base
from tensorflow.python.trackable import layer_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class Mapping(TrackableDataStructure, collections_abc.Mapping):
    """An append-only trackable mapping data structure with string keys.

  Maintains checkpoint dependencies on its contents (which must also be
  trackable), named based on its keys.

  Note that once a key has been added, it may not be deleted or replaced.
  """

    def __init__(self, *args, **kwargs):
        """Construct a new sequence. Arguments are passed to `dict()`."""
        super().__init__()
        self._storage = self._make_storage(*args, **kwargs)
        self._storage.update({key: self._track_value(value, name=self._name_element(key)) for key, value in self._storage.items()})

    def __copy__(self):
        return type(self)(copy.copy(self._storage))

    def __deepcopy__(self, memo):
        return type(self)(copy.deepcopy(self._storage, memo))

    def _make_storage(self, *args, **kwargs):
        return dict(*args, **kwargs)

    @property
    def _values(self):
        """Collect values for TrackableDataStructure."""
        ordered = list(zip(*sorted(self.items(), key=lambda it: it[0])))
        if ordered:
            return ordered[1]
        return []

    def _name_element(self, key):
        if not isinstance(key, str):
            raise TypeError(f'Mapping accepts only string keys, but got a key {repr(key)}.')
        return str(key)

    def __setitem__(self, key, value):
        name = self._name_element(key)
        value = self._track_value(value, name=name)
        current_value = self._storage.setdefault(key, value)
        if current_value is not value:
            raise ValueError(f"Mappings are an append-only data structure. Tried to overwrite the key '{key}' with value {value}, but it already contains {current_value}")

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def __getitem__(self, key):
        return self._storage[key]

    def __len__(self):
        return len(self._storage)

    def __repr__(self):
        return 'Mapping(%s)' % (repr(self._storage),)

    def __iter__(self):
        return iter(self._storage)