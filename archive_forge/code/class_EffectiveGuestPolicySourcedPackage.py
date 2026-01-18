from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EffectiveGuestPolicySourcedPackage(_messages.Message):
    """A guest policy package including its source.

  Fields:
    package: A software package to configure on the VM instance.
    source: Name of the guest policy providing this config.
  """
    package = _messages.MessageField('Package', 1)
    source = _messages.StringField(2)