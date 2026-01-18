from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def ConvertToEnum(versioned_expr, messages):
    """Converts a string version of a versioned expr to the enum representation.

  Args:
    versioned_expr: string, string version of versioned expr to convert.
    messages: messages, The set of available messages.

  Returns:
    A versioned expression enum.
  """
    return messages.SecurityPolicyRuleMatcher.VersionedExprValueValuesEnum(versioned_expr)