from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerTagValuesCreateRequest(_messages.Message):
    """A CloudresourcemanagerTagValuesCreateRequest object.

  Fields:
    tagValue: A TagValue resource to be passed as the request body.
    validateOnly: Optional. Set as true to perform the validations necessary
      for creating the resource, but not actually perform the action.
  """
    tagValue = _messages.MessageField('TagValue', 1)
    validateOnly = _messages.BooleanField(2)