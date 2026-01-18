from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OnDeployUpdatePolicy(_messages.Message):
    """Security patches are only applied when a function is redeployed.

  Fields:
    runtimeVersion: Output only. contains the runtime version which was used
      during latest function deployment.
  """
    runtimeVersion = _messages.StringField(1)