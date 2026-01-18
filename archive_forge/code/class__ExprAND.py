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
class _ExprAND(_ExprLogical):
    """AND node.

  AND with left-to-right shortcut pruning.
  """

    def Evaluate(self, obj):
        if not self._left.Evaluate(obj):
            return False
        if not self._right.Evaluate(obj):
            return False
        return True