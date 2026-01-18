from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.core.document_renderers import renderer
def Entities(self, buf):
    """Escapes special characters to their entity tags.

    This is applied after font embellishments.

    Args:
      buf: Normal text that may contain special characters.

    Returns:
      The string with special characters converted to entity tags.
    """
    esc = re.sub("(``[^`]*)''", '\\1&acute;&acute;', buf)
    return esc.replace('...', '&hellip;')