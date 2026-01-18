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
def Highlight(text, terms, stylize=None):
    """Stylize desired terms in a string.

  Returns a copy of the original string with all substrings matching the given
  terms (with case-insensitive matching) stylized.

  Args:
    text: str, the original text to be highlighted.
    terms: [str], a list of terms to be matched.
    stylize: callable, the function to use to stylize the terms.

  Returns:
    str, the highlighted text.
  """
    if stylize is None:
        stylize = _Stylize
    for term in filter(bool, terms):
        matches = re.finditer(term, text, re.IGNORECASE)
        match_strings = set([text[match.start():match.end()] for match in matches])
        for match_string in match_strings:
            text = text.replace(match_string, stylize(match_string))
    return text