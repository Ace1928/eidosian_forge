import datetime
import json
import math
import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
class DogeMsg(messages.Message):
    such_string = messages.StringField(1)
    wow = messages.IntegerField(2, variant=messages.Variant.INT64)
    very_unsigned = messages.IntegerField(3, variant=messages.Variant.UINT64)
    much_repeated = messages.IntegerField(4, variant=messages.Variant.INT64, repeated=True)