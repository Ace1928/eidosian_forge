from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigManagementConfigSyncError(_messages.Message):
    """Errors pertaining to the installation of Config Sync

  Fields:
    errorMessage: A string representing the user facing error message
  """
    errorMessage = _messages.StringField(1)