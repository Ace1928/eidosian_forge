from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Standard(_messages.Message):
    """Standard represents the standard deployment strategy.

  Fields:
    postdeploy: Optional. Configuration for the postdeploy job. If this is not
      configured, postdeploy job will not be present.
    predeploy: Optional. Configuration for the predeploy job. If this is not
      configured, predeploy job will not be present.
    verify: Whether to verify a deployment.
  """
    postdeploy = _messages.MessageField('Postdeploy', 1)
    predeploy = _messages.MessageField('Predeploy', 2)
    verify = _messages.BooleanField(3)