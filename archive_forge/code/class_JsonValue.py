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
class JsonValue(messages.Message):
    """Any valid JSON value."""
    is_null = messages.BooleanField(1, default=False)
    boolean_value = messages.BooleanField(2)
    string_value = messages.StringField(3)
    double_value = messages.FloatField(4, variant=messages.Variant.DOUBLE)
    integer_value = messages.IntegerField(5, variant=messages.Variant.INT64)
    object_value = messages.MessageField('JsonObject', 6)
    array_value = messages.MessageField('JsonArray', 7)