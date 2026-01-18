import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
class ExtendedEnumValueDescriptor(messages.Message):
    """Enum value descriptor with additional fields.

    Fields:
      name: Name of enumeration value.
      number: Number of enumeration value.
      description: Description of this enum value.
    """
    name = messages.StringField(1)
    number = messages.IntegerField(2, variant=messages.Variant.INT32)
    description = messages.StringField(100)