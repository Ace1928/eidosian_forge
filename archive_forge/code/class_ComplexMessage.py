import pickle
import re
import sys
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
class ComplexMessage(messages.Message):
    a3 = messages.IntegerField(3)
    b1 = messages.StringField(1)
    c2 = messages.StringField(2)