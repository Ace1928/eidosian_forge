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
def _IsMessageFieldSpecified(specified_fields, message_field):
    """Get api fields of arguments when at least one is specified.

  Args:
    specified_fields: List[str], list of api fields that have been specified.
    message_field: str, message field we are determining if specified

  Returns:
    bool, whether the message field is specified.
  """
    for specified_field in specified_fields:
        if specified_field.startswith(message_field):
            return True
    else:
        return False