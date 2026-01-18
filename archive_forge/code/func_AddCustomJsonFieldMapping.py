import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def AddCustomJsonFieldMapping(message_type, python_name, json_name, package=None):
    """Add a custom wire encoding for a given message field.

    This is primarily used in generated code, to handle enum values
    which happen to be Python keywords.

    Args:
      message_type: (messages.Message) A message type
      python_name: (basestring) Python name for this value.
      json_name: (basestring) JSON name to be used on the wire.
      package: (NoneType, optional) No effect, exists for legacy compatibility.
    """
    if not issubclass(message_type, messages.Message):
        raise exceptions.TypecheckError('Cannot set JSON field mapping for non-message "%s"' % message_type)
    try:
        _ = message_type.field_by_name(python_name)
    except KeyError:
        raise exceptions.InvalidDataError('Field %s not recognized for type %s' % (python_name, message_type))
    field_mappings = _JSON_FIELD_MAPPINGS.setdefault(message_type, {})
    _CheckForExistingMappings('field', message_type, python_name, json_name)
    field_mappings[python_name] = json_name