from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShieldedVmIntegrityPolicy(_messages.Message):
    """The policy describes the baseline against which VM instance boot
  integrity is measured.

  Fields:
    updateAutoLearnPolicy: Updates the integrity policy baseline using the
      measurements from the VM instance's most recent boot.
  """
    updateAutoLearnPolicy = _messages.BooleanField(1)