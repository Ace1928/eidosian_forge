from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AttachedPlatformVersionInfo(_messages.Message):
    """Information about a supported Attached Clusters platform version.

  Fields:
    version: Platform version name.
  """
    version = _messages.StringField(1)