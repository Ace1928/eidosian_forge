from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.core.document_renderers import renderer
def LinkGlobalFlags(self, line):
    """Add global flags links to line if any.

    Args:
      line: The text line.

    Returns:
      line with annoted global flag links.
    """
    return re.sub('(--[-a-z]+)', '<a href="/#\\1">\\1</a>', line)