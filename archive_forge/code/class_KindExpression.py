from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KindExpression(_messages.Message):
    """A representation of a kind.

  Fields:
    name: The name of the kind.
  """
    name = _messages.StringField(1)