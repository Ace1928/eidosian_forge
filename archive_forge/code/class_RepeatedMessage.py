import cgi
import datetime
import inspect
import os
import re
import socket
import types
import unittest
import six
from six.moves import range  # pylint: disable=redefined-builtin
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
class RepeatedMessage(messages.Message):
    """Contains all message types as repeated fields."""

    class SimpleEnum(messages.Enum):
        """Simple enumeration type."""
        VAL1 = 1
        VAL2 = 2
    double_value = messages.FloatField(1, variant=messages.Variant.DOUBLE, repeated=True)
    float_value = messages.FloatField(2, variant=messages.Variant.FLOAT, repeated=True)
    int64_value = messages.IntegerField(3, variant=messages.Variant.INT64, repeated=True)
    uint64_value = messages.IntegerField(4, variant=messages.Variant.UINT64, repeated=True)
    int32_value = messages.IntegerField(5, variant=messages.Variant.INT32, repeated=True)
    bool_value = messages.BooleanField(6, variant=messages.Variant.BOOL, repeated=True)
    string_value = messages.StringField(7, variant=messages.Variant.STRING, repeated=True)
    bytes_value = messages.BytesField(8, variant=messages.Variant.BYTES, repeated=True)
    enum_value = messages.EnumField(SimpleEnum, 10, repeated=True)