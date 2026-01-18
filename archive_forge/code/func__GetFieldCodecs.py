import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def _GetFieldCodecs(field, attr):
    result = [getattr(_CUSTOM_FIELD_CODECS.get(field), attr, None), getattr(_FIELD_TYPE_CODECS.get(type(field)), attr, None)]
    return [x for x in result if x is not None]