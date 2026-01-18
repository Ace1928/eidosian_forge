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
def _ParseTypeFromStr(arg_type, data):
    """Parses type from string.

  Args:
    arg_type: str, string representation of type
    data: dict, raw argument data

  Returns:
    The type to use as argparse accepts it.
  """
    if arg_type == ARG_OBJECT:
        return ArgObject.FromData(data)
    elif arg_type == ARG_LIST:
        return Hook.FromPath('googlecloudsdk.calliope.arg_parsers:ArgList:')
    elif (builtin_type := BUILTIN_TYPES.get(arg_type, None)):
        return builtin_type
    else:
        return Hook.FromPath(arg_type)