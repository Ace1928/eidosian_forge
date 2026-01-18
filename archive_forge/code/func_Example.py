from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.core.document_renderers import renderer
def Example(self, line):
    """Displays line as an indented example.

    Args:
      line: The example line.
    """
    self.Blank()
    if not self._example:
        self._example = True
        self._fill = 2
        self._out.write('<p><code>\n')
    indent = len(line)
    line = line.lstrip()
    indent -= len(line)
    self._out.write('&nbsp;' * (self._fill + indent))
    self._out.write(line)
    self._out.write('<br>\n')