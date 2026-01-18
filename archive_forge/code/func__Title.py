from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.core.document_renderers import renderer
def _Title(self):
    """Renders an HTML document title."""
    self._out.write('<html>\n<head>\n')
    if self._title:
        self._out.write('<title>%s</title>\n' % self._title)
    self._out.write('<style>\n  code { color: green; }\n</style>\n<script>\n  window.onload = function() {\n    if (parent.navigation.navigate) {\n      parent.navigation.navigate(document.location.href);\n    }\n  }\n</script>\n')