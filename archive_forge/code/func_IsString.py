from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
def IsString(self, name, peek=False):
    """Skips leading space and checks if the next token is name.

    One of space, '(', or end of input terminates the next token.

    Args:
      name: The token name to check.
      peek: Does not consume the string on match if True.

    Returns:
      True if the next space or ( separated token is name.
    """
    if not self.SkipSpace():
        return False
    i = self.GetPosition()
    if not self._expr[i:].startswith(name):
        return False
    i += len(name)
    if self.EndOfInput(i) or self._expr[i].isspace() or self._expr[i] == '(':
        if not peek:
            self.SetPosition(i)
        return True
    return False