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
@classmethod
def ToChoiceMap(cls, choices):
    """Converts a list of choices into a map for easy value lookup.

    Args:
      choices: [Choice], The choices.

    Returns:
      {arg_value: enum_value}, A mapping of user input to the value that should
      be used. All arg_values have already been converted to lowercase for
      comparison.
    """
    if not choices:
        return {}
    return {c.arg_value: c.enum_value for c in choices}