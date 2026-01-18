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
def PrintNameSection(self, disable_header=False):
    """Prints the command line name section.

    Args:
      disable_header: Disable printing the section header if True.
    """
    if not disable_header:
        self.PrintSectionHeader('NAME')
    self._out('{command} - {index}\n'.format(command=self._command_name, index=_GetIndexFromCapsule(self._capsule)))