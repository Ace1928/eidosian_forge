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
class _AdditionalPropsType(_FieldSpecType):
    """Type converts string into list of apitools message instances for map field.

  Type function returns a list of apitools messages with key, value fields ie
  [Message(key=key1, value=value1), Message(key=key2, value=value2), etc].
  The list of messages is how apitools specifies map fields.

  Attributes:
    key_spec: _FieldSpecType, specifes expected type of key field
    value_spec: _FieldSpecType, specifies expected type of value field
  """

    def __init__(self, key_spec, value_spec, **kwargs):
        super(_AdditionalPropsType, self).__init__(**kwargs)
        self.key_spec = key_spec
        self.value_spec = value_spec

    def __call__(self, arg_value):
        parsed_arg_value = self.arg_type(arg_value)
        messages = []
        for k, v in sorted(parsed_arg_value.items()):
            message_instance = self.field.type()
            self.key_spec.ParseIntoMessage(message_instance, k)
            self.value_spec.ParseIntoMessage(message_instance, v)
            messages.append(message_instance)
        return messages