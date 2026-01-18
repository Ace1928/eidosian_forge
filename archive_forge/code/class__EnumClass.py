import types
import weakref
import six
from apitools.base.protorpclite import util
class _EnumClass(_DefinitionClass):
    """Meta-class used for defining the Enum base class.

    Meta-class enables very specific behavior for any defined Enum
    class.  All attributes defined on an Enum sub-class must be integers.
    Each attribute defined on an Enum sub-class is translated
    into an instance of that sub-class, with the name of the attribute
    as its name, and the number provided as its value.  It also ensures
    that only one level of Enum class hierarchy is possible.  In other
    words it is not possible to delcare sub-classes of sub-classes of
    Enum.

    This class also defines some functions in order to restrict the
    behavior of the Enum class and its sub-classes.  It is not possible
    to change the behavior of the Enum class in later classes since
    any new classes may be defined with only integer values, and no methods.
    """

    def __init__(cls, name, bases, dct):
        if not (bases == (object,) or bases == (Enum,)):
            raise EnumDefinitionError('Enum type %s may only inherit from Enum' % name)
        cls.__by_number = {}
        cls.__by_name = {}
        if bases != (object,):
            for attribute, value in dct.items():
                if attribute in _RESERVED_ATTRIBUTE_NAMES:
                    continue
                if not isinstance(value, six.integer_types):
                    raise EnumDefinitionError('May only use integers in Enum definitions.  Found: %s = %s' % (attribute, value))
                if value < 0:
                    raise EnumDefinitionError('Must use non-negative enum values.  Found: %s = %d' % (attribute, value))
                if value > MAX_ENUM_VALUE:
                    raise EnumDefinitionError('Must use enum values less than or equal %d.  Found: %s = %d' % (MAX_ENUM_VALUE, attribute, value))
                if value in cls.__by_number:
                    raise EnumDefinitionError('Value for %s = %d is already defined: %s' % (attribute, value, cls.__by_number[value].name))
                instance = object.__new__(cls)
                cls.__init__(instance, attribute, value)
                cls.__by_name[instance.name] = instance
                cls.__by_number[instance.number] = instance
                setattr(cls, attribute, instance)
        _DefinitionClass.__init__(cls, name, bases, dct)

    def __iter__(cls):
        """Iterate over all values of enum.

        Yields:
          Enumeration instances of the Enum class in arbitrary order.
        """
        return iter(cls.__by_number.values())

    def names(cls):
        """Get all names for Enum.

        Returns:
          An iterator for names of the enumeration in arbitrary order.
        """
        return cls.__by_name.keys()

    def numbers(cls):
        """Get all numbers for Enum.

        Returns:
          An iterator for all numbers of the enumeration in arbitrary order.
        """
        return cls.__by_number.keys()

    def lookup_by_name(cls, name):
        """Look up Enum by name.

        Args:
          name: Name of enum to find.

        Returns:
          Enum sub-class instance of that value.
        """
        return cls.__by_name[name]

    def lookup_by_number(cls, number):
        """Look up Enum by number.

        Args:
          number: Number of enum to find.

        Returns:
          Enum sub-class instance of that value.
        """
        return cls.__by_number[number]

    def __len__(cls):
        return len(cls.__by_name)