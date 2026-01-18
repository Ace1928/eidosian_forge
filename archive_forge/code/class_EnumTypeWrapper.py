class EnumTypeWrapper(object):
    """A utility for finding the names of enum values."""
    DESCRIPTOR = None

    def __init__(self, enum_type):
        """Inits EnumTypeWrapper with an EnumDescriptor."""
        self._enum_type = enum_type
        self.DESCRIPTOR = enum_type

    def Name(self, number):
        """Returns a string containing the name of an enum value."""
        try:
            return self._enum_type.values_by_number[number].name
        except KeyError:
            pass
        if not isinstance(number, int):
            raise TypeError('Enum value for {} must be an int, but got {} {!r}.'.format(self._enum_type.name, type(number), number))
        else:
            raise ValueError('Enum {} has no name defined for value {!r}'.format(self._enum_type.name, number))

    def Value(self, name):
        """Returns the value corresponding to the given enum name."""
        try:
            return self._enum_type.values_by_name[name].number
        except KeyError:
            pass
        raise ValueError('Enum {} has no value defined for name {!r}'.format(self._enum_type.name, name))

    def keys(self):
        """Return a list of the string names in the enum.

    Returns:
      A list of strs, in the order they were defined in the .proto file.
    """
        return [value_descriptor.name for value_descriptor in self._enum_type.values]

    def values(self):
        """Return a list of the integer values in the enum.

    Returns:
      A list of ints, in the order they were defined in the .proto file.
    """
        return [value_descriptor.number for value_descriptor in self._enum_type.values]

    def items(self):
        """Return a list of the (name, value) pairs of the enum.

    Returns:
      A list of (str, int) pairs, in the order they were defined
      in the .proto file.
    """
        return [(value_descriptor.name, value_descriptor.number) for value_descriptor in self._enum_type.values]

    def __getattr__(self, name):
        """Returns the value corresponding to the given enum name."""
        try:
            return super(EnumTypeWrapper, self).__getattribute__('_enum_type').values_by_name[name].number
        except KeyError:
            pass
        raise AttributeError('Enum {} has no value defined for name {!r}'.format(self._enum_type.name, name))