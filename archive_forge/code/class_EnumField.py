import types
import weakref
import six
from apitools.base.protorpclite import util
class EnumField(Field):
    """Field definition for enum values.

    Enum fields may have default values that are delayed until the
    associated enum type is resolved. This is necessary to support
    certain circular references.

    For example:

      class Message1(Message):

        class Color(Enum):

          RED = 1
          GREEN = 2
          BLUE = 3

        # This field default value  will be validated when default is accessed.
        animal = EnumField('Message2.Animal', 1, default='HORSE')

      class Message2(Message):

        class Animal(Enum):

          DOG = 1
          CAT = 2
          HORSE = 3

        # This fields default value will be validated right away since Color
        # is already fully resolved.
        color = EnumField(Message1.Color, 1, default='RED')
    """
    VARIANTS = frozenset([Variant.ENUM])
    DEFAULT_VARIANT = Variant.ENUM

    def __init__(self, enum_type, number, **kwargs):
        """Constructor.

        Args:
          enum_type: Enum type for field.  Must be subclass of Enum.
          number: Number of field.  Must be unique per message class.
          required: Whether or not field is required.  Mutually exclusive to
            'repeated'.
          repeated: Whether or not field is repeated.  Mutually exclusive to
            'required'.
          variant: Wire-format variant hint.
          default: Default value for field if not found in stream.

        Raises:
          FieldDefinitionError when invalid enum_type is provided.
        """
        valid_type = isinstance(enum_type, six.string_types) or (enum_type is not Enum and isinstance(enum_type, type) and issubclass(enum_type, Enum))
        if not valid_type:
            raise FieldDefinitionError('Invalid enum type: %s' % enum_type)
        if isinstance(enum_type, six.string_types):
            self.__type_name = enum_type
            self.__type = None
        else:
            self.__type = enum_type
        super(EnumField, self).__init__(number, **kwargs)

    def validate_default_element(self, value):
        """Validate default element of Enum field.

        Enum fields allow for delayed resolution of default values
        when the type of the field has not been resolved. The default
        value of a field may be a string or an integer. If the Enum
        type of the field has been resolved, the default value is
        validated against that type.

        Args:
          value: Value to validate.

        Raises:
          ValidationError if value is not expected message type.

        """
        if isinstance(value, (six.string_types, six.integer_types)):
            if self.__type:
                self.__type(value)
            return value
        return super(EnumField, self).validate_default_element(value)

    @property
    def type(self):
        """Enum type used for field."""
        if self.__type is None:
            found_type = find_definition(self.__type_name, self.message_definition())
            if not (found_type is not Enum and isinstance(found_type, type) and issubclass(found_type, Enum)):
                raise FieldDefinitionError('Invalid enum type: %s' % found_type)
            self.__type = found_type
        return self.__type

    @property
    def default(self):
        """Default for enum field.

        Will cause resolution of Enum type and unresolved default value.
        """
        try:
            return self.__resolved_default
        except AttributeError:
            resolved_default = super(EnumField, self).default
            if isinstance(resolved_default, (six.string_types, six.integer_types)):
                resolved_default = self.type(resolved_default)
            self.__resolved_default = resolved_default
            return self.__resolved_default