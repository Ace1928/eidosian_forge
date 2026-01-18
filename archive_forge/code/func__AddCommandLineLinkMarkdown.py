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
def _AddCommandLineLinkMarkdown(self, doc):
    """Add $ command ... link markdown to doc."""
    if not self._command_path:
        return doc
    pat = re.compile(self.CommandLineExamplePattern())
    doc = self._LinkMarkdown(doc, pat, with_args=True)
    return doc