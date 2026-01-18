from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllToAllTraffic(_messages.Message):
    """Predefined traffic shape in which each `group` member sends traffic to
  each other `group` member (except self).

  Fields:
    group: List of coordinates participating in the AllToAll traffic exchange.
  """
    group = _messages.MessageField('CoordinateList', 1)