from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShieldedVmIdentity(_messages.Message):
    """A Shielded VM Identity.

  Fields:
    encryptionKey: An Endorsement Key (EK) issued to the Shielded VM's vTPM.
    kind: [Output Only] Type of the resource. Always
      compute#shieldedVmIdentity for shielded VM identity entry.
    signingKey: An Attestation Key (AK) issued to the Shielded VM's vTPM.
  """
    encryptionKey = _messages.MessageField('ShieldedVmIdentityEntry', 1)
    kind = _messages.StringField(2, default='compute#shieldedVmIdentity')
    signingKey = _messages.MessageField('ShieldedVmIdentityEntry', 3)