from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateInfo(_messages.Message):
    """UpdateInfo contains resource (instance groups, etc), status and other
  intermediate information relevant to a node pool upgrade.

  Fields:
    blueGreenInfo: Information of a blue-green upgrade.
  """
    blueGreenInfo = _messages.MessageField('BlueGreenInfo', 1)