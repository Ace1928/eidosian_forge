import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def _DecodeCustomFieldNames(message_type, encoded_message):
    field_remappings = _JSON_FIELD_MAPPINGS.get(message_type, {})
    if field_remappings:
        decoded_message = json.loads(encoded_message)
        for python_name, json_name in list(field_remappings.items()):
            if json_name in decoded_message:
                decoded_message[python_name] = decoded_message.pop(json_name)
        encoded_message = json.dumps(decoded_message)
    return encoded_message