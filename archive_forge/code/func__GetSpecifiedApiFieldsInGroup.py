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
def _GetSpecifiedApiFieldsInGroup(arguments, namespace):
    """Get api fields of arguments when at least arg is specified in namespace.

  Args:
    arguments: List[yaml_arg_schema.YAMLArgument], list of arguments we want
      to see if they are specified.
    namespace: The parsed command line argument namespace.

  Returns:
    List[str] of api_fields that are specified in the namespace.
  """
    specified_fields = []
    for arg in arguments:
        if arg.IsApiFieldSpecified(namespace):
            specified_fields.extend(arg.api_fields)
    return specified_fields