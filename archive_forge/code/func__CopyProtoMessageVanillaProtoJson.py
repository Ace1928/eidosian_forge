import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def _CopyProtoMessageVanillaProtoJson(message):
    codec = protojson.ProtoJson()
    return codec.decode_message(type(message), codec.encode_message(message))