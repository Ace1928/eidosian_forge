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
def GetFieldFromMessage(message, field_path):
    """Extract the field object from the message using a dotted field path.

  If the field does not exist, an error is logged.

  Args:
    message: The apitools message to dig into.
    field_path: str, The dotted path of attributes and sub-attributes.

  Returns:
    The Field object.
  """
    fields = field_path.split('.')
    for f in fields[:-1]:
        message = _GetField(message, f).type
    return _GetField(message, fields[-1])