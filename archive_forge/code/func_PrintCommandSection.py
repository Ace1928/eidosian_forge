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
def PrintCommandSection(self, name, subcommands, is_topic=False, disable_header=False):
    """Prints a group or command section.

    Args:
      name: str, The section name singular form.
      subcommands: dict, The subcommand dict.
      is_topic: bool, True if this is a TOPIC subsection.
      disable_header: Disable printing the section header if True.
    """
    content = ''
    for subcommand, help_info in sorted(six.iteritems(subcommands)):
        if self._is_hidden or not help_info.is_hidden:
            content += '\n*link:{ref}[{cmd}]*::\n\n{txt}\n'.format(ref='/'.join(self._command_path + [subcommand]), cmd=subcommand, txt=help_info.help_text)
    if content:
        if not disable_header:
            self.PrintSectionHeader(name + 'S')
        if is_topic:
            self._out('The supplementary help topics are:\n')
        else:
            self._out('{cmd} is one of the following:\n'.format(cmd=self._UserInput(name)))
        self._out(content)