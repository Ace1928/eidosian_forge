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
def PrintSynopsisSection(self, disable_header=False):
    """Prints the command line synopsis section.

    Args:
      disable_header: Disable printing the section header if True.
    """
    if self.is_topic:
        return
    self._SetArgSections()
    code = base.MARKDOWN_CODE
    em = base.MARKDOWN_ITALIC
    if not disable_header:
        self.PrintSectionHeader('SYNOPSIS')
    self._out('{code}{command}{code}'.format(code=code, command=self._command_name))
    if self._subcommands and self._subgroups:
        self._out(' ' + em + 'GROUP' + em + ' | ' + em + 'COMMAND' + em)
    elif self._subcommands:
        self._out(' ' + em + 'COMMAND' + em)
    elif self._subgroups:
        self._out(' ' + em + 'GROUP' + em)
    remainder_usage = []
    for section in self._arg_sections:
        self._out(' ')
        self._out(usage_text.GetArgUsage(section.args, markdown=True, top=True, remainder_usage=remainder_usage))
    if self._global_flags:
        self._out(' [' + em + self._top.upper() + '_WIDE_FLAG ...' + em + ']')
    if remainder_usage:
        self._out(' ')
        self._out(' '.join(remainder_usage))
    self._out('\n')