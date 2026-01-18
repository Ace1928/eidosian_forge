from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import pkgutil
import textwrap
from googlecloudsdk import api_lib
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _AppendParagraph(lines):
    """Appends paragraph markdown to lines.

  Paragraph markdown is used to add paragraphs in nested lists at the list
  prevaling indent. _AppendParagraph does not append the markdown if the last
  line in lines is already a paragraph markdown.

  A line containing only the + character is a paragraph markdown. It renders
  a blank line and starts the next paragraph of lines using the prevailing
  indent. A blank line would also start a new paragraph but would decrease the
  prevailing indent.

  Args:
    lines: The lines to append to.
  """
    if lines and (not lines[-1].endswith('\n+\n')):
        if lines[-1].endswith('\n'):
            lines.append('+\n')
        else:
            lines.append('\n+\n')