import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
class MessageWithFields(messages.Message):
    field1 = messages.IntegerField(10)
    field2 = messages.StringField(30)
    field3 = messages.IntegerField(20)