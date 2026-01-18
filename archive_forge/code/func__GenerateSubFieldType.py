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
def _GenerateSubFieldType(self, message, api_field, is_label_field=False):
    """Retrieves the the type of the field from messsage.

    Args:
      message: Apitools message class
      api_field: str, field path of message
      is_label_field: bool, whether field is part of labels map field

    Returns:
      _FieldSpecType, Type function that returns apitools message
        instance or list of instances from string value.
    """
    f = arg_utils.GetFieldFromMessage(message, api_field)
    arg_obj = self._GetFieldTypeFromSpec(api_field)
    return arg_obj.GenerateType(f, is_label_field=is_label_field, is_root=False)