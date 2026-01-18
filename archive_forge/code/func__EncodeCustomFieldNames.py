import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def _EncodeCustomFieldNames(message, encoded_value):
    field_remappings = list(_JSON_FIELD_MAPPINGS.get(type(message), {}).items())
    if field_remappings:
        decoded_value = json.loads(encoded_value)
        for python_name, json_name in field_remappings:
            if python_name in encoded_value:
                decoded_value[json_name] = decoded_value.pop(python_name)
        encoded_value = json.dumps(decoded_value)
    return encoded_value