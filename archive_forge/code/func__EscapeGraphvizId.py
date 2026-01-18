from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import exceptions
import six
def _EscapeGraphvizId(name):
    """Escape a string for use as in Graphviz.

  Args:
    name: The string to escape.

  Returns:
    The `name', with double-quotes escaped, and quotes around it.

  Raises:
    exceptions.UnsupportedNameException: If the name is incompatible with
      Graphviz ID escaping.
  """
    if name.endswith('\\'):
        raise exceptions.UnsupportedNameException('Unsupported name for Graphviz ID escaping: {0!r}'.format(name))
    return '"{0}"'.format(name.replace('"', '\\"'))