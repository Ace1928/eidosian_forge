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
def ClearUnspecifiedMutexFields(message, namespace, arg_group):
    """Clears message fields associated with this mutex ArgGroup.

  Clearing fields is necessary when using read_modify_update. This prevents
  more than one field in a mutex group from being sent in a request message.
  Apitools does not contain information on which fields are mutually exclusive.
  Therefore, we use the api_fields in the argument group to determine which
  fields should be mutually exclusive.

  Args:
    message: The api message that needs to have fields cleared
    namespace: The parsed command line argument namespace.
    arg_group: yaml_arg_schema.ArgGroup, arg
  """
    if not arg_group.mutex or not arg_group.IsApiFieldSpecified(namespace):
        return
    arg_api_fields = arg_group.api_fields
    arg_group_api_field = _GetSharedParent(arg_api_fields)
    first_child_fields = _GetFirstChildFields(arg_api_fields, shared_parent=arg_group_api_field)
    specified_fields = _GetSpecifiedApiFieldsInGroup(arg_group.arguments, namespace)
    for api_field in first_child_fields:
        if not _IsMessageFieldSpecified(specified_fields, api_field):
            ResetFieldInMessage(message, api_field)