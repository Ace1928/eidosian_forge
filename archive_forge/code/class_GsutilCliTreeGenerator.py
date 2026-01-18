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
class GsutilCliTreeGenerator(CliTreeGenerator):
    """gsutil CLI tree generator."""

    def __init__(self, *args, **kwargs):
        super(GsutilCliTreeGenerator, self).__init__(*args, **kwargs)
        self.topics = []

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
        return encoding.Decode(output)

    def AddFlags(self, command, content, is_global=False):
        """Adds flags in content lines to command."""

        def _Add(name, description):
            value = ''
            type_ = 'bool'
            default = ''
            command[cli_tree.LOOKUP_FLAGS][name] = _Flag(name=name, description=description, type_=type_, value=value, default=default, is_required=False, is_global=is_global)
        parse = re.compile(' *((-[^ ]*,)* *(-[^ ]*) *)(.*)')
        name = None
        description = []
        for line in content:
            if line.startswith('  -'):
                if name:
                    _Add(name, '\n'.join(description))
                match = parse.match(line)
                name = match.group(3)
                description = [match.group(4).rstrip()]
            elif len(line) > 16:
                description.append(line[16:].rstrip())
        if name:
            _Add(name, '\n'.join(description))

    def SubTree(self, path):
        """Generates and returns the CLI subtree rooted at path."""
        command = _Command(path)
        is_help_command = len(path) > 1 and path[1] == 'help'
        if is_help_command:
            cmd = path[1:]
        else:
            cmd = path[1:] + ['--help']
        text = self.Run(cmd)
        collector = _GsutilCollector(text)
        while True:
            heading, content = collector.Collect()
            if not heading:
                break
            elif heading == 'CAPSULE':
                if content:
                    command[cli_tree.LOOKUP_CAPSULE] = content[0].split('-', 1)[1].strip()
            elif heading == 'COMMANDS':
                if is_help_command:
                    continue
                for line in content:
                    try:
                        name = line.split()[0]
                    except IndexError:
                        continue
                    if name == 'update':
                        continue
                    command[cli_tree.LOOKUP_IS_GROUP] = True
                    command[cli_tree.LOOKUP_COMMANDS][name] = self.SubTree(path + [name])
            elif heading == 'FLAGS':
                self.AddFlags(command, content)
            elif heading == 'SYNOPSIS':
                commands = []
                for line in content:
                    if not line:
                        break
                    cmd = line.split()
                    if len(cmd) <= len(path):
                        continue
                    if cmd[:len(path)] == path:
                        name = cmd[len(path)]
                        if name[0].islower() and name not in ('off', 'on', 'false', 'true'):
                            commands.append(name)
                if len(commands) > 1:
                    command[cli_tree.LOOKUP_IS_GROUP] = True
                    for name in commands:
                        command[cli_tree.LOOKUP_COMMANDS][name] = self.SubTree(path + [name])
            elif heading == 'TOPICS':
                for line in content:
                    try:
                        self.topics.append(line.split()[0])
                    except IndexError:
                        continue
            elif heading.isupper():
                if heading.lower() == path[-1]:
                    heading = 'DESCRIPTION'
                command[cli_tree.LOOKUP_SECTIONS][heading] = '\n'.join([line[2:] for line in content])
        return command

    def Generate(self):
        """Generates and returns the CLI tree rooted at self.command_name."""
        tree = self.SubTree([self.command_name])
        text = self.Run(['help', 'options'])
        collector = _GsutilCollector(text)
        while True:
            heading, content = collector.Collect()
            if not heading:
                break
            if heading == 'FLAGS':
                self.AddFlags(tree, content, is_global=True)
        help_command = tree[cli_tree.LOOKUP_COMMANDS]['help']
        help_command[cli_tree.LOOKUP_IS_GROUP] = True
        for topic in self.topics:
            help_command[cli_tree.LOOKUP_COMMANDS][topic] = self.SubTree(help_command[cli_tree.LOOKUP_PATH] + [topic])
        tree[cli_tree.LOOKUP_CLI_VERSION] = self.GetVersion()
        tree[cli_tree.LOOKUP_VERSION] = cli_tree.VERSION
        return tree