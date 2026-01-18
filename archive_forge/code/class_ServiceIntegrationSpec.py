from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceIntegrationSpec(_messages.Message):
    """Specifies the parameters to configure an integration with instances.

  Fields:
    backupDr: A ServiceIntegrationSpecBackupDRSpec attribute.
  """
    backupDr = _messages.MessageField('ServiceIntegrationSpecBackupDRSpec', 1)