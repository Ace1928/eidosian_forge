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
class _ExprGlobal(_Expr):
    """Global restriction function call node.

  Attributes:
    _call: The function call object.
  """

    def __init__(self, backend, call):
        super(_ExprGlobal, self).__init__(backend)
        self._call = call

    def Evaluate(self, obj):
        return self._call.Evaluate(obj)