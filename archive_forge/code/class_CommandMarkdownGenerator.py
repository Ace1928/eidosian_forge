from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
import re
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
class CommandMarkdownGenerator(MarkdownGenerator):
    """Command help markdown document generator.

  Attributes:
    _command: The CommandCommon instance for command.
    _root_command: The root CLI command instance.
    _subcommands: The dict of subcommand help indexed by subcommand name.
    _subgroups: The dict of subgroup help indexed by subcommand name.
  """

    def __init__(self, command):
        """Constructor.

    Args:
      command: A calliope._CommandCommon instance. Help is extracted from this
        calliope command, group or topic.
    """
        self._command = command
        command.LoadAllSubElements()
        self._root_command = command._TopCLIElement()
        super(CommandMarkdownGenerator, self).__init__(command.GetPath(), command.ReleaseTrack(), command.IsHidden())
        self._capsule = self._command.short_help
        self._docstring = self._command.long_help
        self._ExtractSectionsFromDocstring(self._docstring)
        self._sections['description'] = self._sections.get('DESCRIPTION', '')
        self._sections.update(getattr(self._command, 'detailed_help', {}))
        self._subcommands = command.GetSubCommandHelps()
        self._subgroups = command.GetSubGroupHelps()
        self._sort_top_level_args = command.ai.sort_args

    def _SetSectionHelp(self, name, lines):
        """Sets section name help composed of lines.

    Args:
      name: The section name.
      lines: The list of lines in the section.
    """
        while lines and (not lines[0]):
            lines = lines[1:]
        while lines and (not lines[-1]):
            lines = lines[:-1]
        if lines:
            self._sections[name] = '\n'.join(lines)

    def _ExtractSectionsFromDocstring(self, docstring):
        """Extracts section help from the command docstring."""
        name = 'DESCRIPTION'
        lines = []
        for line in textwrap.dedent(docstring).strip().splitlines():
            if len(line) >= 4 and line.startswith('## '):
                self._SetSectionHelp(name, lines)
                name = line[3:]
                lines = []
            else:
                lines.append(line)
        self._SetSectionHelp(name, lines)

    def IsValidSubPath(self, sub_command_path):
        """Returns True if the given sub command path is valid from this node."""
        return self._root_command.IsValidSubPath(sub_command_path)

    def GetArguments(self):
        """Returns the command arguments."""
        return self._command.ai.arguments

    def GetNotes(self):
        """Returns the explicit and auto-generated NOTES section contents."""
        return self._command.GetNotesHelpSection(self._sections.get('NOTES'))