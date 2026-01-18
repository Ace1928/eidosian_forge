from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import copy
import difflib
import enum
import io
import re
import sys
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import arg_parsers_usage_text
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import util as format_util
import six
def GetUsage(command, argument_interceptor):
    """Return the command Usage string.

  Args:
    command: calliope._CommandCommon, The command object that we're helping.
    argument_interceptor: parser_arguments.ArgumentInterceptor, the object that
      tracks all of the flags for this command or group.

  Returns:
    str, The command usage string.
  """
    command.LoadAllSubElements()
    command_path = ' '.join(command.GetPath())
    topic = len(command.GetPath()) >= 2 and command.GetPath()[1] == 'topic'
    command_id = 'topic' if topic else 'command'
    buf = io.StringIO()
    buf.write('Usage: ')
    usage_parts = []
    if not topic:
        usage_parts.append(GetArgUsage(argument_interceptor, brief=True, optional=False, top=True))
    group_helps = command.GetSubGroupHelps()
    command_helps = command.GetSubCommandHelps()
    groups = sorted((name for name, help_info in six.iteritems(group_helps) if command.IsHidden() or not help_info.is_hidden))
    commands = sorted((name for name, help_info in six.iteritems(command_helps) if command.IsHidden() or not help_info.is_hidden))
    all_subtypes = []
    if groups:
        all_subtypes.append('group')
    if commands:
        all_subtypes.append(command_id)
    if groups or commands:
        usage_parts.append('<%s>' % ' | '.join(all_subtypes))
        optional_flags = None
    else:
        optional_flags = GetFlags(argument_interceptor, optional=True)
    usage_msg = ' '.join(usage_parts)
    non_option = '{command} '.format(command=command_path)
    buf.write(non_option + usage_msg + '\n')
    if groups:
        WrapWithPrefix('group may be', ' | '.join(groups), HELP_INDENT, LINE_WIDTH, spacing='  ', writer=buf)
    if commands:
        WrapWithPrefix('%s may be' % command_id, ' | '.join(commands), HELP_INDENT, LINE_WIDTH, spacing='  ', writer=buf)
    if optional_flags:
        WrapWithPrefix('optional flags may be', ' | '.join(optional_flags), HELP_INDENT, LINE_WIDTH, spacing='  ', writer=buf)
    buf.write('\n' + GetHelpHint(command))
    return buf.getvalue()