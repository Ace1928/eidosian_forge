from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AttachedClusterUser(_messages.Message):
    """Identities of a user-type subject for Attached clusters.

  Fields:
    username: Required. The name of the user, e.g. `my-gcp-id@gmail.com`.
  """
    username = _messages.StringField(1)