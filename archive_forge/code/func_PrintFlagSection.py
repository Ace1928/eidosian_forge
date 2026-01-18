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
def PrintFlagSection(self, heading, arg, disable_header=False):
    """Prints a flag section.

    Args:
      heading: The flag section heading name.
      arg: The flag args / group.
      disable_header: Disable printing the section header if True.
    """
    if not disable_header:
        self.PrintSectionHeader(heading, sep=False)
    self._PrintArgGroup(arg)