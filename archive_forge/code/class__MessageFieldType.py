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
class _MessageFieldType(_FieldSpecType):
    """Type that converts string input into apitools message.

  Attributes:
    field_specs: list[_FieldSpecType], list of message's fields
  """

    def __init__(self, field_specs, **kwargs):
        super(_MessageFieldType, self).__init__(**kwargs)
        self.field_specs = field_specs

    def _ParseFieldsIntoMessage(self, arg_value):
        """Iterates through fields and adds fields to message instance."""
        message_instance = self.field.type()
        for arg_type in self.field_specs:
            value = arg_value.get(arg_type.arg_name)
            arg_type.ParseIntoMessage(message_instance, value)
        return message_instance

    def __call__(self, arg_value):
        """Converts string into apitools message."""
        parsed_arg_value = self.arg_type(arg_value)
        if isinstance(parsed_arg_value, list):
            return [self._ParseFieldsIntoMessage(r) for r in parsed_arg_value]
        else:
            return self._ParseFieldsIntoMessage(parsed_arg_value)