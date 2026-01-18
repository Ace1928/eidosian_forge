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
def ResetFieldInMessage(message, field_path):
    """Resets the given field in the message object.

  Args:
    message: A constructed apitools message object to inject the value into.
    field_path: str, The dotted path of attributes and sub-attributes.
  """
    if not message:
        return
    sub_message = message
    fields = field_path.split('.')
    for f in fields[:-1]:
        sub_message = getattr(sub_message, f, None)
        if not sub_message:
            break
    else:
        sub_message.reset(fields[-1])