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
def SkipSpace(self, token=None, terminators=''):
    """Skips spaces in the expression string.

    Args:
      token: The expected next token description string, None if end of input is
        OK. This string is used in the exception message. It is not used to
        validate the type of the next token.
      terminators: Space characters in this string will not be skipped.

    Raises:
      ExpressionSyntaxError: End of input reached after skipping and a token is
        expected.

    Returns:
      True if the expression is not at end of input.
    """
    while not self.EndOfInput():
        c = self._expr[self._position]
        if not c.isspace() or c in terminators:
            return True
        self._position += 1
    if token:
        raise resource_exceptions.ExpressionSyntaxError('{0} expected [{1}].'.format(token, self.Annotate()))
    return False