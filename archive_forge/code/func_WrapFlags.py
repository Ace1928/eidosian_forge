from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.document_renderers import devsite_scripts
from googlecloudsdk.core.document_renderers import html_renderer
def WrapFlags(self, tag, match_regex, css_classes):
    """Wraps all regex matches from example in tag with classes."""
    matches = [m.span() for m in re.finditer(match_regex, self._whole_example)]
    wrapped_example = ''
    left = 0
    for match_left, match_right in matches:
        wrapped_example += self._whole_example[left:match_left]
        wrapped_example += '<' + tag + ' class="' + ' '.join(css_classes) + '">'
        wrapped_example += self._whole_example[match_left:match_right]
        wrapped_example += '</' + tag + '>'
        left = match_right
    wrapped_example += self._whole_example[left:]
    return wrapped_example