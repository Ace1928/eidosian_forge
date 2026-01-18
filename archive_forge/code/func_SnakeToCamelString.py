from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import re
from apitools.base.protorpclite import messages as proto_messages
from apitools.base.py import encoding as apitools_encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import files
import six
def SnakeToCamelString(snake):
    """Change a snake_case string into a camelCase string.

  Args:
    snake: str, the string to be transformed.

  Returns:
    str, the transformed string.
  """
    parts = snake.split('_')
    if not parts:
        return snake
    leading_blanks = 0
    for p in parts:
        if not p:
            leading_blanks += 1
        else:
            break
    if leading_blanks:
        parts = parts[leading_blanks:]
        if not parts:
            return '_' * (leading_blanks - 1)
        parts[0] = '_' * leading_blanks + parts[0]
    return ''.join(parts[:1] + [s.capitalize() for s in parts[1:]])