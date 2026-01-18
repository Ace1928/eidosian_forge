from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import re
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import http_encoding
import six
def SetFieldInMessage(message, field_path, value):
    """Sets the given field in the message object.

  Args:
    message: A constructed apitools message object to inject the value into.
    field_path: str, The dotted path of attributes and sub-attributes.
    value: The value to set.
  """
    fields = field_path.split('.')
    for f in fields[:-1]:
        sub_message = getattr(message, f)
        is_repeated = _GetField(message, f).repeated
        if not sub_message:
            sub_message = _GetField(message, f).type()
            if is_repeated:
                sub_message = [sub_message]
            setattr(message, f, sub_message)
        message = sub_message[0] if is_repeated else sub_message
    field_type = _GetField(message, fields[-1]).type
    if isinstance(value, dict):
        value = encoding.PyValueToMessage(field_type, value)
    if isinstance(value, list):
        for i, item in enumerate(value):
            if isinstance(field_type, type) and (not isinstance(item, field_type)):
                value[i] = encoding.PyValueToMessage(field_type, item)
    setattr(message, fields[-1], value)