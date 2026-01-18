from __future__ import annotations
import collections
from collections.abc import Sequence
from typing import Any
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import cli
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.run import flags
def AddContainerFlags(parser: parser_arguments.ArgumentInterceptor, container_arg_group: calliope_base.ArgumentGroup):
    """AddContainerFlags updates parser to add --container arg parsing.

  Args:
    parser: The parser to patch.
    container_arg_group: Arguments that can be specified per-container.
  """
    flags.ContainerFlag().AddToParser(parser)
    container_arg_group.AddToParser(parser)
    container_parser = ContainerParser(parser.parser, container_arg_group)
    parser.parser.parse_known_args = container_parser.ParseKnownArgs