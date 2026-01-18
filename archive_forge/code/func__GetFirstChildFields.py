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
def _GetFirstChildFields(api_fields, shared_parent=None):
    """Gets first child for api_fields.

  For a list of fields, supply the full api_field up through the first child.
  For example:
      ['a.b.c', 'a.b.d.e.f'] with shared parent 'a.b'
      returns children ['a.b.c', 'a.b.d']

  Args:
    api_fields: [str], list of api fields to get children from
    shared_parent: str | None, the shared parent between all api fields

  Returns:
    [str], list of the children api_fields
  """
    start_index = len(shared_parent) + 1 if shared_parent else 0
    child_fields = []
    for api_field in api_fields:
        if shared_parent and (not api_field.startswith(shared_parent)):
            raise ValueError('Invalid parent: {} does not start with {}.'.format(api_field, shared_parent))
        children = api_field[start_index:].split('.')
        first_child = children and children[0]
        if shared_parent and first_child:
            field = '.'.join((shared_parent, first_child))
        else:
            field = shared_parent or first_child
        if field:
            child_fields.append(field)
    return child_fields