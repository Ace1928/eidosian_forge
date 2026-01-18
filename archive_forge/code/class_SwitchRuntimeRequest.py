from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SwitchRuntimeRequest(_messages.Message):
    """Request for switching a Managed Notebook Runtime.

  Fields:
    acceleratorConfig: accelerator config.
    machineType: machine type.
    requestId: Idempotent request UUID.
  """
    acceleratorConfig = _messages.MessageField('RuntimeAcceleratorConfig', 1)
    machineType = _messages.StringField(2)
    requestId = _messages.StringField(3)