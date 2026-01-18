from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MoveImpact(_messages.Message):
    """A message to group impacts of moving the target resource.

  Fields:
    detail: User friendly impact detail in a free form message.
  """
    detail = _messages.StringField(1)