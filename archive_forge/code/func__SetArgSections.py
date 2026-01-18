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
def _SetArgSections(self):
    """Sets self._arg_sections in document order."""
    if self._arg_sections is None:
        self._arg_sections, self._global_flags = usage_text.GetArgSections(self.GetArguments(), self.is_root, self.is_group, self.sort_top_level_args)