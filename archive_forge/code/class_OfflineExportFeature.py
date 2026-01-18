from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OfflineExportFeature(_messages.Message):
    """An offline export transfer, where data is downloaded onto the appliance
  at Google and copied from the appliance at the customer site.

  Fields:
    source: Required. The source of the transfer.
    stsAccount: Output only. The Storage Transfer Service service account used
      for this transfer. This account must be given roles/storage.admin access
      to the output bucket.
    transferManifest: Manifest file for the transfer. When this is provided,
      only files specificied in the manifest file are transferred. Only one
      GCSSource can be set with this option.
    workloadAccount: Output only. The service account associated with this
      transfer. This account must be granted the roles/storage.admin role on
      the output bucket, and the roles/cloudkms.cryptoKeyDecrypter and
      roles/cloudkms.publicKeyViewer roles on the customer managed key.
  """
    source = _messages.MessageField('GcsSource', 1, repeated=True)
    stsAccount = _messages.StringField(2)
    transferManifest = _messages.MessageField('TransferManifest', 3)
    workloadAccount = _messages.StringField(4)