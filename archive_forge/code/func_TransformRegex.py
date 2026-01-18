from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
def TransformRegex(r, expression, does_match=None, nomatch=''):
    """Returns does_match or r itself if r matches expression, nomatch otherwise.

  Args:
    r: A String.
    expression: expression to apply to r.
    does_match: If the string matches expression then return _does_match_
      otherwise return the string itself if _does_match_ is not defined.
    nomatch: Returns this value if the string does not match expression.

  Returns:
    does_match or r if r matches expression, nomatch otherwise.
  """
    if not r:
        return nomatch
    try:
        if expression:
            match_re = re.compile(expression)
            if match_re.match(r):
                return does_match or r
    except (AttributeError, TypeError, ValueError):
        pass
    return nomatch