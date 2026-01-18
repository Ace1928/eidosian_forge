import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def _SafeDecodeBytes(unused_field, value):
    """Decode the urlsafe base64 value into bytes."""
    try:
        result = base64.urlsafe_b64decode(str(value))
        complete = True
    except TypeError:
        result = value
        complete = False
    return CodecResult(value=result, complete=complete)