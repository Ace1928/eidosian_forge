from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AzureClusterGroup(_messages.Message):
    """Identities of a group-type subject for Azure clusters.

  Fields:
    group: Required. The name of the group, e.g. `my-group@domain.com`.
  """
    group = _messages.StringField(1)