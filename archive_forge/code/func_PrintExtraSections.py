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
def PrintExtraSections(self, disable_header=False):
    """Print extra sections not in excluded_sections.

    Extra sections are sections that have not been printed yet.
    PrintSectionIfExists() skips sections that have already been printed.

    Args:
      disable_header: Disable printing the section header if True.
    """
    excluded_sections = set(self._final_sections + ['NOTES', 'UNIVERSE ADDITIONAL INFO'])
    for section in sorted(self._sections):
        if section.isupper() and section not in excluded_sections:
            self.PrintSectionIfExists(section, disable_header=disable_header)