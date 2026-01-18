from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def EntrypointFlags():
    """Encapsulate flags for customizing container command."""
    args_flag = StringListFlag('--args', metavar='ARG', help="Comma-separated arguments passed to the command run by the container image. If not specified and no '--command' is provided, the container image's default Cmd is used. Otherwise, if not specified, no arguments are passed. To reset this field to its default, pass an empty string.")
    command_flag = StringListFlag('--command', metavar='COMMAND', help="Entrypoint for the container image. If not specified, the container image's default Entrypoint is run. To reset this field to its default, pass an empty string.")
    return FlagGroup(args_flag, command_flag)