import base64
import datetime
import json
import sys
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
class MessageWithRemappings(messages.Message):

    class SomeEnum(messages.Enum):
        enum_value = 1
        second_value = 2
    enum_field = messages.EnumField(SomeEnum, 1)
    double_encoding = messages.EnumField(SomeEnum, 2)
    another_field = messages.StringField(3)
    repeated_enum = messages.EnumField(SomeEnum, 4, repeated=True)
    repeated_field = messages.StringField(5, repeated=True)