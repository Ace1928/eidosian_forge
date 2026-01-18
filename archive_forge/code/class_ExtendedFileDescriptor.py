import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
class ExtendedFileDescriptor(messages.Message):
    """File descriptor with additional fields.

    Fields:
      package: Fully qualified name of package that definitions belong to.
      message_types: Message definitions contained in file.
      enum_types: Enum definitions contained in file.
      description: Description of this file.
      additional_imports: Extra imports used in this package.
    """
    package = messages.StringField(2)
    message_types = messages.MessageField(ExtendedMessageDescriptor, 4, repeated=True)
    enum_types = messages.MessageField(ExtendedEnumDescriptor, 5, repeated=True)
    description = messages.StringField(100)
    additional_imports = messages.StringField(101, repeated=True)