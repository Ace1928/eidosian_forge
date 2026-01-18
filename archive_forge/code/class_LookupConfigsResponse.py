from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class LookupConfigsResponse(_messages.Message):
    """Response with assigned configs for the instance.

  Fields:
    apt: Configs for apt.
    goo: Configs for windows.
    windowsUpdate: Configs for Windows Update.
    yum: Configs for yum.
    zypper: Configs for Zypper.
  """
    apt = _messages.MessageField('AptPackageConfig', 1)
    goo = _messages.MessageField('GooPackageConfig', 2)
    windowsUpdate = _messages.MessageField('WindowsUpdateConfig', 3)
    yum = _messages.MessageField('YumPackageConfig', 4)
    zypper = _messages.MessageField('ZypperPackageConfig', 5)