from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SkippedDetail(_messages.Message):
    """Details for an outcome with a SKIPPED outcome summary.

  Fields:
    incompatibleAppVersion: If the App doesn't support the specific API level.
    incompatibleArchitecture: If the App doesn't run on the specific
      architecture, for example, x86.
    incompatibleDevice: If the requested OS version doesn't run on the
      specific device model.
  """
    incompatibleAppVersion = _messages.BooleanField(1)
    incompatibleArchitecture = _messages.BooleanField(2)
    incompatibleDevice = _messages.BooleanField(3)