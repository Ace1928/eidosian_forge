from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetReference(_messages.Message):
    """A TargetReference object.

  Fields:
    target: A string attribute.
  """
    target = _messages.StringField(1)