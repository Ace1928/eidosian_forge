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
def ParseKnownArgs(self, args: Sequence[Any], namespace: parser_extensions.Namespace) -> tuple[parser_extensions.Namespace, Sequence[Any]]:
    """Performs custom --container arg parsing.

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
    """
    remaining = []
    containers = collections.defaultdict(list)
    current = remaining
    i = 0
    while i < len(args):
        value = args[i]
        i += 1
        if value == self._CONTAINER_FLAG_NAME:
            if i >= len(args):
                remaining.append(value)
            else:
                current = containers[args[i]]
                i += 1
        elif isinstance(value, str) and value.startswith(self._CONTAINER_FLAG_NAME + '='):
            current = containers[value.split(sep='=', maxsplit=1)[1]]
        elif value == '--':
            remaining.append(value)
            remaining.extend(args[i:])
            break
        else:
            current.append(value)
    if not containers:
        return self._parse_known_args(args=remaining, namespace=namespace)
    namespace.containers = {}
    namespace._specified_args['containers'] = self._CONTAINER_FLAG_NAME
    for container_name, container_args in containers.items():
        container_namespace = parser_extensions.Namespace()
        container_namespace = self._NewContainerParser().parse_args(args=container_args, namespace=container_namespace)
        namespace.containers[container_name] = container_namespace
    namespace, unknown_args = self._parse_known_args(args=remaining, namespace=namespace)
    self._CheckForContainerFlags(namespace)
    return (namespace, unknown_args)