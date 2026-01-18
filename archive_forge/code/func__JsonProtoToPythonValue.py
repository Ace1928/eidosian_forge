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
def _JsonProtoToPythonValue(json_proto):
    util.Typecheck(json_proto, _JSON_PROTO_TYPES)
    return _JSON_PROTO_TO_PYTHON_MAP[type(json_proto)](json_proto)