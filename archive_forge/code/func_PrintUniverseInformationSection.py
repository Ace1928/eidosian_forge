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
def PrintUniverseInformationSection(self, disable_header=False):
    """Prints the command line information section.

    The information section provides disclaimer information on whether a command
    is available in a particular universe domain.

    Args:
      disable_header: Disable printing the section header if True.
    """
    if properties.IsDefaultUniverse():
        return
    if not disable_header:
        self.PrintSectionHeader('INFORMATION')
    code = base.MARKDOWN_CODE
    em = base.MARKDOWN_ITALIC
    if self._command.IsUniverseCompatible():
        info_body = f'{code}{self._command_name}{code} is supported in universe domain {em}{properties.GetUniverseDomain()}{em}; however, some of the values used in the help text may not be available. Command examples may not work as-is and may requires changes before execution.'
    else:
        info_body = f'{code}{self._command_name}{code} is not available in universe domain {em}{properties.GetUniverseDomain()}{em}.'
    self._out(info_body)
    self.PrintSectionIfExists('UNIVERSE ADDITIONAL INFO')