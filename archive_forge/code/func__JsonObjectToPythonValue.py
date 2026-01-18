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
def _JsonObjectToPythonValue(json_value):
    util.Typecheck(json_value, JsonObject)
    return dict([(prop.key, _JsonValueToPythonValue(prop.value)) for prop in json_value.properties])