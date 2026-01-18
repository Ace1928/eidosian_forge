import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def _GetField(message, field_path):
    for field in field_path:
        if field not in dir(message):
            raise KeyError('no field "%s"' % field)
        message = getattr(message, field)
    return message