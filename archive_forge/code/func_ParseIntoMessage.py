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
def ParseIntoMessage(self, message_instance, value):
    """Sets field in a message after value is parsed into correct type.

    Args:
      message_instance: apitools message instance we are parsing value into
      value: value we are parsing into apitools message
    """
    if value is None and self.repeated:
        field_value = []
    else:
        field_value = value
    arg_utils.SetFieldInMessage(message_instance, self.api_field, field_value)