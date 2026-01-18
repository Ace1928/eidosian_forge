from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DriverRunner(_messages.Message):
    """Driver runner configuration.

  Fields:
    masterDriverRunner: Optional. (default) Run the driver on the master node.
    yarnDriverRunner: Optional. Run the driver on worker nodes using YARN.
  """
    masterDriverRunner = _messages.MessageField('MasterDriverRunner', 1)
    yarnDriverRunner = _messages.MessageField('YarnDriverRunner', 2)