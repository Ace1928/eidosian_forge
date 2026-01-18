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
def PrintFinalSections(self, disable_header=False):
    """Print the final sections in order.

    Args:
      disable_header: Disable printing the section header if True.
    """
    for section in self._final_sections:
        self.PrintSectionIfExists(section, disable_header=disable_header)
    self.PrintNotesSection(disable_header=disable_header)