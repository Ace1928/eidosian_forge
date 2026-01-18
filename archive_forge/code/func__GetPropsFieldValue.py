from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import arg_parsers_usage_text as usage_text
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
import six
def _GetPropsFieldValue(self, field):
    """Retrieves AdditionalProperties field value.

    Args:
      field: apitools instance that contains AdditionalProperties field

    Returns:
      list of apitools AdditionalProperties messages.
    """
    if not field:
        return []
    if self._is_list_field:
        return field
    return arg_utils.GetFieldValueFromMessage(field, arg_utils.ADDITIONAL_PROPS)