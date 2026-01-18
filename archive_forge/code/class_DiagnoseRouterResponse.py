from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiagnoseRouterResponse(_messages.Message):
    """DiagnoseRouterResponse contains the current status for a specific
  router.

  Fields:
    result: The network status of a specific router.
    updateTime: The time when the router status was last updated.
  """
    result = _messages.MessageField('RouterStatus', 1)
    updateTime = _messages.StringField(2)