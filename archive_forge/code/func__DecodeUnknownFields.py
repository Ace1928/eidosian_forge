import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def _DecodeUnknownFields(message, encoded_message):
    """Rewrite unknown fields in message into message.destination."""
    destination = _UNRECOGNIZED_FIELD_MAPPINGS.get(type(message))
    if destination is None:
        return message
    pair_field = message.field_by_name(destination)
    if not isinstance(pair_field, messages.MessageField):
        raise exceptions.InvalidDataFromServerError('Unrecognized fields must be mapped to a compound message type.')
    pair_type = pair_field.message_type
    if isinstance(pair_type.value, messages.MessageField):
        new_values = _DecodeUnknownMessages(message, json.loads(encoded_message), pair_type)
    else:
        new_values = _DecodeUnrecognizedFields(message, pair_type)
    setattr(message, destination, new_values)
    setattr(message, '_Message__unrecognized_fields', {})
    return message