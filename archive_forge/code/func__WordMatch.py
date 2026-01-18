from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
import unicodedata
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import times
import six
def _WordMatch(backend, key, op, warned_attribute, value, pattern):
    """Applies _MatchOneWordInText to determine if value matches pattern.

  Both value and operand can be lists.

  Args:
    backend: The parser backend object.
    key: The parsed expression key.
    op: The expression operator string.
    warned_attribute: Deprecation warning Boolean attribute name.
    value: The key value or list of values.
    pattern: Pattern value or list of values.

  Returns:
    True if the value (or any element in value if it is a list) matches pattern
    (or any element in operand if it is a list).
  """
    if isinstance(value, dict):
        warned_attribute = None
        values = []
        if value:
            values.extend(six.iterkeys(value))
            values.extend(six.itervalues(value))
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = [value]
    if isinstance(pattern, (list, tuple)):
        patterns = pattern
    else:
        patterns = {pattern}
    for v in values:
        for p in patterns:
            if _MatchOneWordInText(backend, key, op, warned_attribute, v, p):
                return True
    return False