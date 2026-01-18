from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShieldedInstanceConfig(_messages.Message):
    """A set of Shielded Instance options.

  Fields:
    enableIntegrityMonitoring: Optional. Whether to enable integrity
      monitoring.
    enableSecureBoot: Optional. Whether to enable secure boot.
    enableVtpm: Optional. Whether to enable VTPM.
  """
    enableIntegrityMonitoring = _messages.BooleanField(1)
    enableSecureBoot = _messages.BooleanField(2)
    enableVtpm = _messages.BooleanField(3)