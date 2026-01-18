from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprExprCreateList(_messages.Message):
    """A list creation expression. Lists may either be homogenous, e.g. `[1, 2,
  3]`, or heterogeneous, e.g. `dyn([1, 'hello', 2.0])`

  Fields:
    elements: The elements part of the list.
    optionalIndices: The indices within the elements list which are marked as
      optional elements. When an optional-typed value is present, the value it
      contains is included in the list. If the optional-typed value is absent,
      the list element is omitted from the CreateList result.
  """
    elements = _messages.MessageField('GoogleApiExprExpr', 1, repeated=True)
    optionalIndices = _messages.IntegerField(2, repeated=True, variant=_messages.Variant.INT32)