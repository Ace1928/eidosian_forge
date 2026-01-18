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
def _GetNestedGroup(buf, i, beg, end):
    """Returns the index in buf of the end of the nested beg...end group.

  Args:
    buf: Input buffer.
    i: The buf[] index of the first beg character.
    beg: The group begin character.
    end: The group end character.

  Returns:
    The index in buf of the end of the nested beg...end group, 0 if there is
    no group.
  """
    if buf[i] != beg:
        return 0
    nesting = 0
    while i < len(buf):
        if buf[i] == beg:
            nesting += 1
        elif buf[i] == end:
            nesting -= 1
            if nesting <= 0:
                return i
        i += 1
    return 0