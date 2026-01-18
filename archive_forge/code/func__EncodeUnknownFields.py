import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def _EncodeUnknownFields(message):
    """Remap unknown fields in message out of message.source."""
    source = _UNRECOGNIZED_FIELD_MAPPINGS.get(type(message))
    if source is None:
        return message
    result = _CopyProtoMessageVanillaProtoJson(message)
    pairs_field = message.field_by_name(source)
    if not isinstance(pairs_field, messages.MessageField):
        raise exceptions.InvalidUserInputError('Invalid pairs field %s' % pairs_field)
    pairs_type = pairs_field.message_type
    value_field = pairs_type.field_by_name('value')
    value_variant = value_field.variant
    pairs = getattr(message, source)
    codec = _ProtoJsonApiTools.Get()
    for pair in pairs:
        encoded_value = codec.encode_field(value_field, pair.value)
        result.set_unrecognized_field(pair.key, encoded_value, value_variant)
    setattr(result, source, [])
    return result