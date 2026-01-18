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
class _ExprOR(_ExprLogical):
    """OR node.

  OR with left-to-right shortcut pruning.
  """

    def Evaluate(self, obj):
        if self._left.Evaluate(obj):
            return True
        if self._right.Evaluate(obj):
            return True
        return False