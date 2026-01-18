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
Performs custom --container arg parsing.

    Groups arguments after each --container flag to be parsed into that
    container's namespace. For each container a new parser is used to parse that
    container's flags into fresh namespace and those namespaces are stored as a
    dict in namespace.containers. Remaining args are parsed by the orignal
    parser's parse_known_args method.

    Args:
      args: The arguments to parse.
      namespace: The namespace to store parsed args in.

    Returns:
      A tuple containing the updated namespace and a list of unknown args.
    