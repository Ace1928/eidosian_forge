from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerTagValuesDeleteRequest(_messages.Message):
    """A CloudresourcemanagerTagValuesDeleteRequest object.

  Fields:
    etag: Optional. The etag known to the client for the expected state of the
      TagValue. This is to be used for optimistic concurrency.
    name: Required. Resource name for TagValue to be deleted in the format
      tagValues/456.
    validateOnly: Optional. Set as true to perform the validations necessary
      for deletion, but not actually perform the action.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    validateOnly = _messages.BooleanField(3)