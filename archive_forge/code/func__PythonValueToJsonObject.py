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
def _PythonValueToJsonObject(py_value):
    util.Typecheck(py_value, dict)
    return JsonObject(properties=[JsonObject.Property(key=key, value=_PythonValueToJsonValue(value)) for key, value in py_value.items()])