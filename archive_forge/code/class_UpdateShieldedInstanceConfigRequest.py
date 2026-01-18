from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateShieldedInstanceConfigRequest(_messages.Message):
    """Request for updating the Shielded Instance config for a notebook
  instance. You can only use this method on a stopped instance

  Fields:
    shieldedInstanceConfig: ShieldedInstance configuration to be updated.
  """
    shieldedInstanceConfig = _messages.MessageField('ShieldedInstanceConfig', 1)