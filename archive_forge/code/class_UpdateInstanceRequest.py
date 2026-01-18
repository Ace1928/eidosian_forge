from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateInstanceRequest(_messages.Message):
    """The request for UpdateInstance.

  Fields:
    fieldMask: Required. A mask specifying which fields in Instance should be
      updated. The field mask must always be specified; this prevents any
      future fields in Instance from being erased accidentally by clients that
      do not know about them.
    instance: Required. The instance to update, which must always include the
      instance name. Otherwise, only fields mentioned in field_mask need be
      included.
  """
    fieldMask = _messages.StringField(1)
    instance = _messages.MessageField('Instance', 2)