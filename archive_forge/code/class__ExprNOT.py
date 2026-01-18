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
class _ExprNOT(_Expr):
    """NOT node."""

    def __init__(self, backend, expr):
        super(_ExprNOT, self).__init__(backend)
        self._expr = expr

    def Evaluate(self, obj):
        return not self._expr.Evaluate(obj)