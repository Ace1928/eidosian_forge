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
def PrintPositionalsAndFlagsSections(self, disable_header=False):
    """Prints the positionals and flags sections.

    Args:
      disable_header: Disable printing the section header if True.
    """
    if self.is_topic:
        return
    self._SetArgSections()
    for section in self._arg_sections:
        self.PrintFlagSection(section.heading, section.args, disable_header=disable_header)
    if self._global_flags:
        if not disable_header:
            self.PrintSectionHeader('{} WIDE FLAGS'.format(self._top.upper()), sep=False)
        self._out('\nThese flags are available to all commands: {}.\n\nRun *$ {} help* for details.\n'.format(', '.join(sorted(self._global_flags)), self._top))