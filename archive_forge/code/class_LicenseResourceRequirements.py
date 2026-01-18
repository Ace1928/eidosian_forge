from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LicenseResourceRequirements(_messages.Message):
    """A LicenseResourceRequirements object.

  Fields:
    minGuestCpuCount: Minimum number of guest cpus required to use the
      Instance. Enforced at Instance creation and Instance start.
    minMemoryMb: Minimum memory required to use the Instance. Enforced at
      Instance creation and Instance start.
  """
    minGuestCpuCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    minMemoryMb = _messages.IntegerField(2, variant=_messages.Variant.INT32)