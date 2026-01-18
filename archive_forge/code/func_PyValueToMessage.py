import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def PyValueToMessage(message_type, value):
    """Convert the given python value to a message of type message_type."""
    return JsonToMessage(message_type, json.dumps(value))