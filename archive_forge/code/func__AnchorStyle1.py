from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import re
import sys
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.document_renderers import devsite_renderer
from googlecloudsdk.core.document_renderers import html_renderer
from googlecloudsdk.core.document_renderers import linter_renderer
from googlecloudsdk.core.document_renderers import man_renderer
from googlecloudsdk.core.document_renderers import markdown_renderer
from googlecloudsdk.core.document_renderers import renderer
from googlecloudsdk.core.document_renderers import text_renderer
def _AnchorStyle1(self, buf, i):
    """Checks for link:target[text] hyperlink anchor markdown.

    Hyperlink anchors are of the form:
      <link> ':' <target> [ '[' <text> ']' ]
    For example:
      http://www.google.com[Google Search]
    The underlying renderer determines how the parts are displayed.

    Args:
      buf: Input buffer.
      i: The buf[] index of ':'.

    Returns:
      (i, back, target, text)
        i: The buf[] index just past the link, 0 if no link.
        back: The number of characters to retain before buf[i].
        target: The link target.
        text: The link text.
    """
    if i >= 3 and buf[i - 3:i] == 'ftp':
        back = 3
        target_beg = i - 3
    elif i >= 4 and buf[i - 4:i] == 'http':
        back = 4
        target_beg = i - 4
    elif i >= 4 and buf[i - 4:i] == 'link':
        back = 4
        target_beg = i + 1
    elif i >= 5 and buf[i - 5:i] == 'https':
        back = 5
        target_beg = i - 5
    elif i >= 6 and buf[i - 6:i] == 'mailto':
        back = 6
        target_beg = i - 6
    else:
        return (0, 0, None, None)
    text_beg = 0
    text_end = 0
    while True:
        if i >= len(buf) or buf[i].isspace():
            if buf[i - 1] == '.':
                i -= 1
            target_end = i
            text_beg = i
            text_end = i - 1
            break
        if buf[i] == '[':
            target_end = i
            text_beg = i + 1
            text_end = _GetNestedGroup(buf, i, '[', ']')
            break
        if buf[i] in '{}()<>\'"`*':
            break
        i += 1
    if not text_end:
        return (0, 0, None, None)
    return (text_end + 1, back, buf[target_beg:target_end], buf[text_beg:text_end])