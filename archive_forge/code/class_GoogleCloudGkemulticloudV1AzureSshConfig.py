from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AzureSshConfig(_messages.Message):
    """SSH configuration for Azure resources.

  Fields:
    authorizedKey: Required. The SSH public key data for VMs managed by
      Anthos. This accepts the authorized_keys file format used in OpenSSH
      according to the sshd(8) manual page.
  """
    authorizedKey = _messages.StringField(1)