import datetime
import json
import numbers
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import encoding_helper as encoding
from apitools.base.py import exceptions
from apitools.base.py import util
def _JsonToJsonValue(json_data, unused_decoder=None):
    result = _PythonValueToJsonProto(json.loads(json_data))
    if isinstance(result, JsonValue):
        return result
    elif isinstance(result, JsonObject):
        return JsonValue(object_value=result)
    elif isinstance(result, JsonArray):
        return JsonValue(array_value=result)
    else:
        raise exceptions.InvalidDataError('Malformed JsonValue: %s' % json_data)