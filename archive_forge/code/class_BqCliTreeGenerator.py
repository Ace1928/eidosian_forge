from a man-ish style runtime document.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import os
import re
import shlex
import subprocess
import tarfile
import textwrap
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.static_completion import generate as generate_static
from googlecloudsdk.command_lib.static_completion import lookup
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
from six.moves import range
class BqCliTreeGenerator(CliTreeGenerator):
    """bq CLI tree generator."""

    def Run(self, cmd):
        """Runs the root command with args given by cmd and returns the output.

    Args:
      cmd: [str], List of arguments to the root command.
    Returns:
      str, Output of the given command.
    """
        try:
            output = subprocess.check_output(self._root_command_args + cmd)
        except subprocess.CalledProcessError as e:
            if e.returncode != 1:
                raise
            output = e.output
        return encoding.Decode(output).replace('bq.py', 'bq')

    def AddFlags(self, command, content, is_global=False):
        """Adds flags in content lines to command."""
        while content:
            line = content.pop(0)
            name, description = line.strip().split(':', 1)
            paragraph = [description.strip()]
            default = ''
            while content and (not content[0].startswith('  --')):
                line = content.pop(0).strip()
                if line.startswith('(default: '):
                    default = line[10:-1]
                else:
                    paragraph.append(line)
            description = ' '.join(paragraph).strip()
            if name.startswith('--[no]'):
                name = '--' + name[6:]
                type_ = 'bool'
                value = ''
            else:
                value = 'VALUE'
                type_ = 'string'
            command[cli_tree.LOOKUP_FLAGS][name] = _Flag(name=name, description=description, type_=type_, value=value, default=default, is_required=False, is_global=is_global)

    def SubTree(self, path):
        """Generates and returns the CLI subtree rooted at path."""
        command = _Command(path)
        command[cli_tree.LOOKUP_IS_GROUP] = True
        text = self.Run(['help'] + path[1:])
        content = text.split('\n')
        while content:
            line = content.pop(0)
            if not line or not line[0].islower():
                continue
            name, text = line.split(' ', 1)
            description = [text.strip()]
            examples = []
            arguments = []
            paragraph = description
            while content and (not content[0] or not content[0][0].islower()):
                line = content.pop(0).strip()
                if line == 'Arguments:':
                    paragraph = arguments
                elif line == 'Examples:':
                    paragraph = examples
                else:
                    paragraph.append(line)
            subcommand = _Command(path + [name])
            command[cli_tree.LOOKUP_COMMANDS][name] = subcommand
            if description:
                subcommand[cli_tree.LOOKUP_SECTIONS]['DESCRIPTION'] = '\n'.join(description)
            if examples:
                subcommand[cli_tree.LOOKUP_SECTIONS]['EXAMPLES'] = '\n'.join(examples)
        return command

    def Generate(self):
        """Generates and returns the CLI tree rooted at self.command_name."""
        tree = self.SubTree([self.command_name])
        text = self.Run(['--help'])
        collector = _BqCollector(text)
        _, content = collector.Collect(strip_headings=True)
        self.AddFlags(tree, content, is_global=True)
        tree[cli_tree.LOOKUP_CLI_VERSION] = self.GetVersion()
        tree[cli_tree.LOOKUP_VERSION] = cli_tree.VERSION
        return tree