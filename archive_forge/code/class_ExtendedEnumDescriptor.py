import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
class ExtendedEnumDescriptor(messages.Message):
    """Enum class descriptor with additional fields.

    Fields:
      name: Name of Enum without any qualification.
      values: Values defined by Enum class.
      description: Description of this enum class.
      full_name: Fully qualified name of this enum class.
      enum_mappings: Mappings from python to JSON names for enum values.
    """

    class JsonEnumMapping(messages.Message):
        """Mapping from a python name to the wire name for an enum."""
        python_name = messages.StringField(1)
        json_name = messages.StringField(2)
    name = messages.StringField(1)
    values = messages.MessageField(ExtendedEnumValueDescriptor, 2, repeated=True)
    description = messages.StringField(100)
    full_name = messages.StringField(101)
    enum_mappings = messages.MessageField('JsonEnumMapping', 102, repeated=True)