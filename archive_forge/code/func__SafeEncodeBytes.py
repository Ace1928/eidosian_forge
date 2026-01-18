import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def _SafeEncodeBytes(field, value):
    """Encode the bytes in value as urlsafe base64."""
    try:
        if field.repeated:
            result = [base64.urlsafe_b64encode(byte) for byte in value]
        else:
            result = base64.urlsafe_b64encode(value)
        complete = True
    except TypeError:
        result = value
        complete = False
    return CodecResult(value=result, complete=complete)