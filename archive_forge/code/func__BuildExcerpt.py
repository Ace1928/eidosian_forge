from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import re
from googlecloudsdk.command_lib.help_search import lookup
from googlecloudsdk.core.document_renderers import render_document
import six
from six.moves import filter
def _BuildExcerpt(text, snips):
    """Helper function to build excerpt using (start, end) tuples.

  Returns a string that combines substrings of the text (text[start:end]),
  joins them with ellipses

  Args:
    text: the text to excerpt from.
    snips: [(int, int)] list of 2-tuples representing start and end places
        to cut text.

  Returns:
    str, the excerpt.
  """
    snippet = '...'.join([text[snip.AsSlice()] for snip in snips])
    if snips:
        if snips[0].start != 0:
            snippet = '...' + snippet
        if snips[-1].end != len(text):
            snippet += '...'
    return snippet