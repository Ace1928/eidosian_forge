from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class When(_messages.Message):
    """Custom condition the request must satisfy.

  Fields:
    expr: CEL expression to be evaluated.
  """
    expr = _messages.StringField(1)