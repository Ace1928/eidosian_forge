from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagementConfig(_messages.Message):
    """Configuration of the cluster management

  Fields:
    fullManagementConfig: Configuration of the full (Autopilot) cluster
      management. Full cluster management is a preview feature.
    standardManagementConfig: Configuration of the standard (GKE) cluster
      management
  """
    fullManagementConfig = _messages.MessageField('FullManagementConfig', 1)
    standardManagementConfig = _messages.MessageField('StandardManagementConfig', 2)