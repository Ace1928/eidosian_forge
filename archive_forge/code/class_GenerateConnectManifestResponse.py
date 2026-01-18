from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenerateConnectManifestResponse(_messages.Message):
    """GenerateConnectManifestResponse contains manifest information for
  installing/upgrading a Connect agent.

  Fields:
    manifest: The ordered list of Kubernetes resources that need to be applied
      to the cluster for GKE Connect agent installation/upgrade.
  """
    manifest = _messages.MessageField('ConnectAgentResource', 1, repeated=True)