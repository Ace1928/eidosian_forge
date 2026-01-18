from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AzureDiskTemplate(_messages.Message):
    """Configuration for Azure Disks.

  Fields:
    sizeGib: Optional. The size of the disk, in GiBs. When unspecified, a
      default value is provided. See the specific reference in the parent
      resource.
  """
    sizeGib = _messages.IntegerField(1, variant=_messages.Variant.INT32)