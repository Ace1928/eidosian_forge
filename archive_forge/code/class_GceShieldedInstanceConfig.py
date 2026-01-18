from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GceShieldedInstanceConfig(_messages.Message):
    """A set of Compute Engine Shielded instance options.

  Fields:
    enableIntegrityMonitoring: Optional. Whether the instance has integrity
      monitoring enabled.
    enableSecureBoot: Optional. Whether the instance has Secure Boot enabled.
    enableVtpm: Optional. Whether the instance has the vTPM enabled.
  """
    enableIntegrityMonitoring = _messages.BooleanField(1)
    enableSecureBoot = _messages.BooleanField(2)
    enableVtpm = _messages.BooleanField(3)