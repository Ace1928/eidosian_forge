from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartialUpdateInstanceRequest(_messages.Message):
    """Request message for BigtableInstanceAdmin.PartialUpdateInstance.

  Fields:
    instance: Required. The Instance which will (partially) replace the
      current value.
    updateMask: Required. The subset of Instance fields which should be
      replaced. Must be explicitly set.
  """
    instance = _messages.MessageField('Instance', 1)
    updateMask = _messages.StringField(2)