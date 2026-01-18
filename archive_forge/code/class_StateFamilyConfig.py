from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StateFamilyConfig(_messages.Message):
    """State family configuration.

  Fields:
    isRead: If true, this family corresponds to a read operation.
    stateFamily: The state family value.
  """
    isRead = _messages.BooleanField(1)
    stateFamily = _messages.StringField(2)