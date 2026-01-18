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
def _ReplaceGDULinksWithUniverseLinks(self, doc):
    """Replace static GDU Links with Universe Links."""
    if self._IsUniverseCompatible():
        doc = re.sub('cloud.google.com', properties.GetUniverseDocumentDomain(), doc)
    return doc