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
def Edit(self, doc=None):
    """Applies edits to a copy of the generated markdown in doc.

    The sub-edit method call order might be significant. This method allows
    the combined edits to be tested without relying on the order.

    Args:
      doc: The markdown document string to edit, None for the output buffer.

    Returns:
      An edited copy of the generated markdown.
    """
    if doc is None:
        doc = self._buf.getvalue()
    doc = self._ExpandFormatReferences(doc)
    doc = self._AddCommandLineLinkMarkdown(doc)
    doc = self._AddCommandLinkMarkdown(doc)
    doc = self._AddManPageLinkMarkdown(doc)
    doc = self._FixAirQuotesMarkdown(doc)
    doc = self._ReplaceGDULinksWithUniverseLinks(doc)
    return doc