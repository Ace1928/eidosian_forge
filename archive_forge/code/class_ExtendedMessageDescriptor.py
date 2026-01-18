import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
class ExtendedMessageDescriptor(messages.Message):
    """Message descriptor with additional fields.

    Fields:
      name: Name of Message without any qualification.
      fields: Fields defined for message.
      message_types: Nested Message classes defined on message.
      enum_types: Nested Enum classes defined on message.
      description: Description of this message.
      full_name: Full qualified name of this message.
      decorators: Decorators to include in the definition when printing.
          Printed in the given order from top to bottom (so the last entry
          is the innermost decorator).
      alias_for: This type is just an alias for the named type.
      field_mappings: Mappings from python to json field names.
    """

    class JsonFieldMapping(messages.Message):
        """Mapping from a python name to the wire name for a field."""
        python_name = messages.StringField(1)
        json_name = messages.StringField(2)
    name = messages.StringField(1)
    fields = messages.MessageField(ExtendedFieldDescriptor, 2, repeated=True)
    message_types = messages.MessageField('extended_descriptor.ExtendedMessageDescriptor', 3, repeated=True)
    enum_types = messages.MessageField(ExtendedEnumDescriptor, 4, repeated=True)
    description = messages.StringField(100)
    full_name = messages.StringField(101)
    decorators = messages.StringField(102, repeated=True)
    alias_for = messages.StringField(103)
    field_mappings = messages.MessageField('JsonFieldMapping', 104, repeated=True)