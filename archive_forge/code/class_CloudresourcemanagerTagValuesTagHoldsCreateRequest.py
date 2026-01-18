from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerTagValuesTagHoldsCreateRequest(_messages.Message):
    """A CloudresourcemanagerTagValuesTagHoldsCreateRequest object.

  Fields:
    parent: Required. The resource name of the TagHold's parent TagValue. Must
      be of the form: `tagValues/{tag-value-id}`.
    tagHold: A TagHold resource to be passed as the request body.
    validateOnly: Optional. Set to true to perform the validations necessary
      for creating the resource, but not actually perform the action.
  """
    parent = _messages.StringField(1, required=True)
    tagHold = _messages.MessageField('TagHold', 2)
    validateOnly = _messages.BooleanField(3)