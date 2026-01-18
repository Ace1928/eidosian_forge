from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceStatusServiceIntegrationStatus(_messages.Message):
    """Represents the status of integration between instance and another
  service. See go/gce-backupdr-design for more details.

  Fields:
    backupDr: A ResourceStatusServiceIntegrationStatusBackupDRStatus
      attribute.
  """
    backupDr = _messages.MessageField('ResourceStatusServiceIntegrationStatusBackupDRStatus', 1)