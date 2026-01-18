import collections
import re
from tensorflow.python.util import tf_inspect
class _PredicateRegistry(object):
    """Registry with predicate-based lookup.

  See the documentation for `register_checkpoint_saver` and
  `register_serializable` for reasons why predicates are required over a
  class-based registry.

  Since this class is used for global registries, each object must be registered
  to unique names (an error is raised if there are naming conflicts). The lookup
  searches the predicates in reverse order, so that later-registered predicates
  are executed first.
  """
    __slots__ = ('_registry_name', '_registered_map', '_registered_predicates', '_registered_names')

    def __init__(self, name):
        self._registry_name = name
        self._registered_map = {}
        self._registered_predicates = {}
        self._registered_names = []

    @property
    def name(self):
        return self._registry_name

    def register(self, package, name, predicate, candidate):
        """Registers a candidate object under the package, name and predicate."""
        if not isinstance(package, str) or not isinstance(name, str):
            raise TypeError(f'The package and name registered to a {self.name} must be strings, got: package={type(package)}, name={type(name)}')
        if not callable(predicate):
            raise TypeError(f'The predicate registered to a {self.name} must be callable, got: {type(predicate)}')
        registered_name = package + '.' + name
        if not _VALID_REGISTERED_NAME.match(registered_name):
            raise ValueError(f"Invalid registered {self.name}. Please check that the package and name follow the regex '{_VALID_REGISTERED_NAME.pattern}': (package='{package}', name='{name}')")
        if registered_name in self._registered_map:
            raise ValueError(f"The name '{registered_name}' has already been registered to a {self.name}. Found: {self._registered_map[registered_name]}")
        self._registered_map[registered_name] = candidate
        self._registered_predicates[registered_name] = predicate
        self._registered_names.append(registered_name)

    def lookup(self, obj):
        """Looks up the registered object using the predicate.

    Args:
      obj: Object to pass to each of the registered predicates to look up the
        registered object.
    Returns:
      The object registered with the first passing predicate.
    Raises:
      LookupError if the object does not match any of the predicate functions.
    """
        return self._registered_map[self.get_registered_name(obj)]

    def name_lookup(self, registered_name):
        """Looks up the registered object using the registered name."""
        try:
            return self._registered_map[registered_name]
        except KeyError:
            raise LookupError(f"The {self.name} registry does not have name '{registered_name}' registered.")

    def get_registered_name(self, obj):
        for registered_name in reversed(self._registered_names):
            predicate = self._registered_predicates[registered_name]
            if predicate(obj):
                return registered_name
        raise LookupError(f'Could not find matching {self.name} for {type(obj)}.')

    def get_predicate(self, registered_name):
        try:
            return self._registered_predicates[registered_name]
        except KeyError:
            raise LookupError(f"The {self.name} registry does not have name '{registered_name}' registered.")

    def get_registrations(self):
        return self._registered_predicates