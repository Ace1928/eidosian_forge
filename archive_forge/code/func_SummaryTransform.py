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
def SummaryTransform(r):
    """A resource transform function to summarize a command search result.

  Uses the "results" attribute of the command to build a summary that includes
  snippets of the help text of the command that include the searched terms.
  Occurrences of the search term will be stylized.

  Args:
    r: a json representation of a command.

  Returns:
    str, a summary of the command.
  """
    summary = GetSummary(r, r[lookup.RESULTS])
    md = io.StringIO(summary)
    rendered_summary = io.StringIO()
    render_document.RenderDocument('text', md, out=rendered_summary, width=len(summary) * 2)
    final_summary = '\n'.join([l.lstrip() for l in rendered_summary.getvalue().splitlines() if l.lstrip()])
    return final_summary