from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerTagValuesPatchRequest(_messages.Message):
    """A CloudresourcemanagerTagValuesPatchRequest object.

  Fields:
    name: Immutable. Resource name for TagValue in the format `tagValues/456`.
    tagValue: A TagValue resource to be passed as the request body.
    updateMask: Optional. Fields to be updated.
    validateOnly: Optional. True to perform validations necessary for updating
      the resource, but not actually perform the action.
  """
    name = _messages.StringField(1, required=True)
    tagValue = _messages.MessageField('TagValue', 2)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)