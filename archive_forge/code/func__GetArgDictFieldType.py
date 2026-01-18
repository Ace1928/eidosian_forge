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
def _GetArgDictFieldType(message, user_field_spec):
    """Retrieves the the type of the field from message.

  Args:
    message: Apitools message class
    user_field_spec: ArgDictFieldSpec, specifies the api field

  Returns:
    _FieldType, type function that returns apitools field class
  """
    field = arg_utils.GetFieldFromMessage(message, user_field_spec.api_field)
    arg_type = user_field_spec.field_type or _GetFieldValueType(field)
    field_spec = _FieldSpec.FromUserData(field, arg_name=user_field_spec.arg_name, api_field=user_field_spec.api_field, required=user_field_spec.required)
    return _FieldType(arg_type=arg_type, field_spec=field_spec, choices=user_field_spec.ChoiceMap())