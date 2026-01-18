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
def _GetSharedParent(api_fields):
    """Gets shared parent of api_fields.

  For a list of fields, find the common parent between them or None.
  For example, ['a.b.c', 'a.b.d'] would return 'a.b'

  Args:
    api_fields: [list], list of api fields that we need to find parent

  Returns:
    str | None, shared common parent or None if one is not found
  """
    if not api_fields:
        return None
    longest_parent = api_fields[0].split('.')
    for field in api_fields:
        substr = field.split('.')
        longest_parent = _GetCommonPrefix(longest_parent, substr)
    return '.'.join(longest_parent) or None