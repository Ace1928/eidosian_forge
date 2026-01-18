import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
def _FixupMessage(self, message_type):
    with self.__DescriptorEnv(message_type):
        for field in message_type.fields:
            if field.field_descriptor.variant == messages.Variant.MESSAGE:
                field_type_name = field.field_descriptor.type_name
                field_type = self.LookupDescriptor(field_type_name)
                if isinstance(field_type, extended_descriptor.ExtendedEnumDescriptor):
                    field.field_descriptor.variant = messages.Variant.ENUM
        for submessage_type in message_type.message_types:
            self._FixupMessage(submessage_type)