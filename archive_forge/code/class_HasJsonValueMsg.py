import datetime
import json
import math
import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
class HasJsonValueMsg(messages.Message):
    some_value = messages.MessageField(extra_types.JsonValue, 1)