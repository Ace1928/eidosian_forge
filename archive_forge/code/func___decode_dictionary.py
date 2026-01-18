import base64
import binascii
import logging
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
def __decode_dictionary(self, message_type, dictionary):
    """Merge dictionary in to message.

        Args:
          message: Message to merge dictionary in to.
          dictionary: Dictionary to extract information from.  Dictionary
            is as parsed from JSON.  Nested objects will also be dictionaries.
        """
    message = message_type()
    for key, value in six.iteritems(dictionary):
        if value is None:
            try:
                message.reset(key)
            except AttributeError:
                pass
            continue
        try:
            field = message.field_by_name(key)
        except KeyError:
            variant = self.__find_variant(value)
            if variant:
                message.set_unrecognized_field(key, value, variant)
            continue
        if field.repeated:
            if not isinstance(value, list):
                value = [value]
            valid_value = [self.decode_field(field, item) for item in value]
            setattr(message, field.name, valid_value)
            continue
        if value == []:
            continue
        try:
            setattr(message, field.name, self.decode_field(field, value))
        except messages.DecodeError:
            if not isinstance(field, messages.EnumField):
                raise
            variant = self.__find_variant(value)
            if variant:
                message.set_unrecognized_field(key, value, variant)
    return message