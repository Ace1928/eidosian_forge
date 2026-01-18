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
def _ParseFieldsIntoMessage(self, arg_value):
    """Iterates through fields and adds fields to message instance."""
    message_instance = self.field.type()
    for arg_type in self.field_specs:
        value = arg_value.get(arg_type.arg_name)
        arg_type.ParseIntoMessage(message_instance, value)
    return message_instance