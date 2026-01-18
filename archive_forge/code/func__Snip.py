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
def _Snip(text, length_per_snippet, terms):
    """Create snippet of text, containing given terms if present.

  The max length of the snippet is the number of terms times the given length.
  This is to prevent a long list of terms from resulting in nonsensically
  short sub-strings. Each substring is up to length given, joined by '...'

  Args:
    text: str, the part of help text to cut. Should be only ASCII characters.
    length_per_snippet: int, the length of the substrings to create containing
        each term.
    terms: [str], the terms to include.

  Returns:
    str, a summary excerpt including the terms, with all consecutive whitespace
        including newlines reduced to a single ' '.
  """
    text = re.sub('\\s+', ' ', text)
    if len(text) <= length_per_snippet:
        return text
    cut_points = [0] + [r.start() for r in re.finditer('\\s', text)] + [len(text)]
    if not terms:
        return _BuildExcerpt(text, [_GetStartAndEnd(None, cut_points, length_per_snippet)])
    unsorted_matches = [re.search(term, text, re.IGNORECASE) for term in terms]
    matches = sorted(filter(bool, unsorted_matches), key=lambda x: x.start())
    snips = []
    for match in matches:
        if not (snips and snips[-1].start < match.start() and (snips[-1].end > match.end())):
            next_slice = _GetStartAndEnd(match, cut_points, length_per_snippet)
            if snips:
                latest = snips[-1]
                if latest.Overlaps(next_slice):
                    latest.Merge(next_slice)
                else:
                    snips.append(next_slice)
            else:
                snips.append(next_slice)
    if not snips:
        snips = [_GetStartAndEnd(None, cut_points, length_per_snippet)]
    return _BuildExcerpt(text, snips)