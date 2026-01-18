class TypeRegistry(object):
    """Provides a type registry for the python registry pattern.

  Contains mappings between types and type specific objects, to implement the
  registry pattern.

  Some example uses of this would be to register different functions depending
  on the type of object.
  """

    def __init__(self):
        self._registry = {}

    def register(self, obj, value):
        """Registers a Python object within the registry.

    Args:
      obj: The object to add to the registry.
      value: The stored value for the 'obj' type.

    Raises:
      KeyError: If the same obj is used twice.
    """
        if obj in self._registry:
            raise KeyError(f'{type(obj)} has already been registered.')
        self._registry[obj] = value

    def lookup(self, obj):
        """Looks up 'obj'.

    Args:
      obj: The object to lookup within the registry.

    Returns:
      Value for 'obj' in the registry if found.
    Raises:
      LookupError: if 'obj' has not been registered.
    """
        for registered in self._registry:
            if isinstance(obj, registered):
                return self._registry[registered]
        raise LookupError(f'{type(obj)} has not been registered.')