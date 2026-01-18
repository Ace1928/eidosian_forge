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
class _DictWrapper(TrackableDataStructure, wrapt.ObjectProxy):
    """Wraps built-in dicts to support restore-on-create for variables.

  _DictWrapper is to Mapping as ListWrapper is to List. Unlike Mapping,
  _DictWrapper allows non-string keys and values and arbitrary mutations (delete
  keys, reassign values). Like ListWrapper, these mutations mean that
  _DictWrapper will raise an exception on save.
  """

    def __init__(self, wrapped_dict=None):
        if wrapped_dict is None:
            wrapped_dict = {}
        if not isinstance(wrapped_dict, collections_abc.Mapping):
            wrapped_dict = dict(wrapped_dict)
        wrapt.ObjectProxy.__init__(self, wrapped_dict)
        TrackableDataStructure.__init__(self)
        self._self_non_string_key = False
        self._self_external_modification = False
        self.__wrapped__.update({key: self._track_value(value, name=self._name_element(key)) for key, value in self.__wrapped__.items()})
        self._update_snapshot()

    def __reduce_ex__(self, protocol):
        return (self.__class__, (self.__wrapped__,))

    def __getattribute__(self, name):
        if hasattr(type(self), name) and isinstance(getattr(type(self), name), property):
            return object.__getattribute__(self, name)
        else:
            return super().__getattribute__(name)

    def copy(self):
        return copy.copy(self)

    def __copy__(self):
        copied = _DictWrapper(copy.copy(self.__wrapped__))
        copied._self_external_modification = self._self_external_modification
        copied._self_non_string_key = self._self_non_string_key
        return copied

    def __deepcopy__(self, memo):
        copied = _DictWrapper(copy.deepcopy(self.__wrapped__, memo))
        copied._self_external_modification = self._self_external_modification
        copied._self_non_string_key = self._self_non_string_key
        return copied

    @property
    def _values(self):
        """Collect values for TrackableDataStructure."""
        ordered = list(zip(*sorted(self.items(), key=lambda it: it[0])))
        if ordered:
            return ordered[1]
        return []

    def _trackable_children(self, save_type=base.SaveType.CHECKPOINT, **kwargs):
        """Check that the object is saveable before listing its dependencies."""
        self._check_self_external_modification()
        if self._self_non_string_key:
            raise ValueError(f"Unable to save the object {self} (a dictionary wrapper constructed automatically on attribute assignment). The wrapped dictionary contains a non-string key which maps to a trackable object or mutable data structure.\n\nIf you don't need this dictionary checkpointed, wrap it in a non-trackable object; it will be subsequently ignored.")
        if self._self_external_modification:
            raise ValueError(f"Unable to save the object {self} (a dictionary wrapper constructed automatically on attribute assignment). The wrapped dictionary was modified outside the wrapper (its final value was {self}, its value when a checkpoint dependency was added was {self._self_last_wrapped_dict_snapshot}), which breaks restoration on object creation.\n\nIf you don't need this dictionary checkpointed, wrap it in a non-trackable object; it will be subsequently ignored.")
        assert not self._dirty
        children = super()._trackable_children(save_type, **kwargs)
        if save_type == base.SaveType.SAVEDMODEL:
            children.update({key: value for key, value in self.items() if _is_function(value)})
        return children

    @property
    def _dirty(self):
        """Check if there has already been a mutation which prevents saving."""
        return self._self_external_modification or self._self_non_string_key

    def _check_self_external_modification(self):
        """Checks for any changes to the wrapped dict not through the wrapper."""
        if self._dirty:
            return
        if self != self._self_last_wrapped_dict_snapshot:
            self._self_external_modification = True
            self._self_last_wrapped_dict_snapshot = None

    def _update_snapshot(self):
        """Acknowledges tracked changes to the wrapped dict."""
        self._attribute_sentinel.invalidate_all()
        if self._dirty:
            return
        self._self_last_wrapped_dict_snapshot = dict(self)

    def _track_value(self, value, name):
        """Allows storage of non-trackable objects."""
        if isinstance(name, str):
            string_key = True
        else:
            name = '-non_string_key'
            string_key = False
        try:
            no_dependency = isinstance(value, NoDependency)
            value = super()._track_value(value=value, name=name)
            if not (string_key or no_dependency):
                self._self_non_string_key = True
            return value
        except ValueError:
            return sticky_attribute_assignment(trackable=self, value=value, name=name)

    def _name_element(self, key):
        """Tells TrackableDataStructure to use keys as names as-is."""
        return key

    def __setitem__(self, key, value):
        """Allow any modifications, but possibly mark the wrapper as unsaveable."""
        self._check_self_external_modification()
        self._maybe_initialize_trackable()
        no_dep = isinstance(value, NoDependency)
        if isinstance(key, str):
            value = self._track_value(value, name=key)
        else:
            value = wrap_or_unwrap(value)
            if not no_dep and isinstance(value, base.Trackable):
                self._self_non_string_key = True
        self.__wrapped__[key] = value
        self._update_snapshot()

    def __delitem__(self, key):
        self._check_self_external_modification()
        del self.__wrapped__[key]
        self._update_snapshot()

    def __repr__(self):
        return 'DictWrapper(%s)' % (repr(self.__wrapped__),)

    def __hash__(self):
        raise TypeError("unhashable type: 'DictWrapper'")

    def __eq__(self, other):
        return self.__wrapped__ == other

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value