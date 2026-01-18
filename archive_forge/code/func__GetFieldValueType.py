from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import unicode_literals
import abc
from collections.abc import Callable
import dataclasses
from typing import Any
from apitools.base.protorpclite import messages as apitools_messages
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import arg_parsers_usage_text as usage_text
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import module_util
def _GetFieldValueType(field):
    """Returns the input type for the apitools field.

  Args:
    field: apitools field instance

  Returns:
    Type function for apitools field input.

  Raises:
    InvalidSchemaError: if the field type is not listed in arg_utils.TYPES
  """
    arg_type = arg_utils.TYPES.get(field.variant)
    if not arg_type:
        raise InvalidSchemaError('Unknown type for field: ' + field.name)
    return arg_type