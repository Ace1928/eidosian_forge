from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OfflineImportFeature(_messages.Message):
    """An offline import transfer, where data is loaded onto the appliance at
  the customer site and ingested at Google.

  Enums:
    StateValueValuesEnum: Output only. The state of the transfer.

  Fields:
    destination: The destination of the transfer.
    state: Output only. The state of the transfer.
    stsAccount: Output only. The Storage Transfer Service service account used
      for this transfer. This account must be given roles/storage.admin access
      to the output bucket.
    transferResults: Output only. The results of the transfer.
    workloadAccount: Output only. The service account associated with this
      transfer. This account must be granted the roles/storage.admin role on
      the output bucket, and the roles/cloudkms.cryptoKeyDecrypter and
      roles/cloudkms.publicKeyViewer roles on the customer managed key, if
      using one.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the transfer.

    Values:
      STATE_UNSPECIFIED: Default value. This value is unused.
      DRAFT: The transfer is associated with a draft order.
      ACTIVE: The appliance used for this transfer has been ordered but is not
        yet back at Google for ingest.
      INGESTING: The data is being ingested off the appliance at Google.
      COMPLETED: The transfer has completed and data is available in the
        output bucket.
      CANCELLED: The transfer has been cancelled and data will not be
        ingested.
    """
        STATE_UNSPECIFIED = 0
        DRAFT = 1
        ACTIVE = 2
        INGESTING = 3
        COMPLETED = 4
        CANCELLED = 5
    destination = _messages.MessageField('GcsDestination', 1)
    state = _messages.EnumField('StateValueValuesEnum', 2)
    stsAccount = _messages.StringField(3)
    transferResults = _messages.MessageField('TransferResults', 4)
    workloadAccount = _messages.StringField(5)