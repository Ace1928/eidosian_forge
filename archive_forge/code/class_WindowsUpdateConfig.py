from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WindowsUpdateConfig(_messages.Message):
    """Configuration settings for the Windows update.

  Fields:
    windowsUpdateServerUri: Optional URI of Windows update server. This sets
      the registry value `WUServer` under
      `HKLM:\\SOFTWARE\\Policies\\Microsoft\\Windows\\WindowsUpdate`.
  """
    windowsUpdateServerUri = _messages.StringField(1)