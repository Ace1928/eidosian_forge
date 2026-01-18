from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.core.document_renderers import renderer
def Link(self, target, text):
    """Renders an anchor.

    Args:
      target: The link target URL.
      text: The text to be displayed instead of the link.

    Returns:
      The rendered link anchor and text.
    """
    if ':' in target or target.startswith('www.'):
        return '<a href="{target}" target=_top>{text}</a>'.format(target=target, text=text or target)
    if '#' in target or target.endswith('..'):
        return '<a href="{target}">{text}</a>'.format(target=target, text=text or target)
    if not text:
        text = target.replace('/', ' ')
    tail = '/help'
    if target.endswith(tail):
        target = target[:-len(tail)]
    target = target.replace('/', '_') + '.html'
    return '<a href="{target}">{text}</a>'.format(target=target, text=text)