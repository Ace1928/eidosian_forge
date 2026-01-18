from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.core.document_renderers import renderer
def _Heading(self, level, heading):
    """Renders an HTML heading.

    Args:
      level: The heading level counting from 1.
      heading: The heading text.
    """
    self._heading = '</dd>\n'
    level += 2
    if level > 9:
        level = 9
    self._out.write('\n<dt><h%d>%s</h%d></dt>\n<dd class="sectionbody">\n' % (level, heading, level))