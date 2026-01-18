from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.core.document_renderers import renderer
def Line(self):
    """Renders a paragraph separating line."""
    self._Flush()
    if not self.HaveBlank():
        self.Blank()
        self._paragraph = True