import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
class CustomField(messages.MessageField):
    """Custom MessageField class."""
    type = int
    message_type = message_types.VoidMessage

    def __init__(self, number, **kwargs):
        super(CustomField, self).__init__(self.message_type, number, **kwargs)

    def value_to_message(self, value):
        return self.message_type()