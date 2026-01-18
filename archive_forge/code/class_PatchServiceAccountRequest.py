from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PatchServiceAccountRequest(_messages.Message):
    """The service account patch request. You can patch only the `display_name`
  and `description` fields. You must use the `update_mask` field to specify
  which of these fields you want to patch. Only the fields specified in the
  request are guaranteed to be returned in the response. Other fields may be
  empty in the response.

  Fields:
    serviceAccount: A ServiceAccount attribute.
    updateMask: A string attribute.
  """
    serviceAccount = _messages.MessageField('ServiceAccount', 1)
    updateMask = _messages.StringField(2)