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
def _CheckForContainerFlags(self, namespace: parser_extensions.Namespace):
    """_CheckForContainerFlags checks that no container flags were specified.

    Args:
      namespace: The namespace to check.
    """
    container_flags = self._GetContainerFlags().intersection(namespace.GetSpecifiedArgNames())
    if container_flags:
        raise parser_errors.ArgumentError('When --container is specified {flags} must be specified after --container.', flags=', '.join(container_flags))