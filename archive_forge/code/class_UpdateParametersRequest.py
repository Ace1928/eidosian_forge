from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateParametersRequest(_messages.Message):
    """Request for UpdateParameters.

  Fields:
    parameters: The parameters to apply to the instance.
    updateMask: Required. Mask of fields to update.
  """
    parameters = _messages.MessageField('MemcacheParameters', 1)
    updateMask = _messages.StringField(2)