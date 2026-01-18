from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerTagValuesTagHoldsDeleteRequest(_messages.Message):
    """A CloudresourcemanagerTagValuesTagHoldsDeleteRequest object.

  Fields:
    name: Required. The resource name of the TagHold to delete. Must be of the
      form: `tagValues/{tag-value-id}/tagHolds/{tag-hold-id}`.
    validateOnly: Optional. Set to true to perform the validations necessary
      for deleting the resource, but not actually perform the action.
  """
    name = _messages.StringField(1, required=True)
    validateOnly = _messages.BooleanField(2)