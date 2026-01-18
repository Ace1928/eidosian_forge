from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateCustomClassRequest(_messages.Message):
    """Request message for the UpdateCustomClass method.

  Fields:
    customClass: Required. The CustomClass to update. The CustomClass's `name`
      field is used to identify the CustomClass to update. Format:
      `projects/{project}/locations/{location}/customClasses/{custom_class}`.
    updateMask: The list of fields to be updated. If empty, all fields are
      considered for update.
    validateOnly: If set, validate the request and preview the updated
      CustomClass, but do not actually update it.
  """
    customClass = _messages.MessageField('CustomClass', 1)
    updateMask = _messages.StringField(2)
    validateOnly = _messages.BooleanField(3)