from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerTagKeysCreateRequest(_messages.Message):
    """A CloudresourcemanagerTagKeysCreateRequest object.

  Fields:
    tagKey: A TagKey resource to be passed as the request body.
    validateOnly: Optional. Set to true to perform validations necessary for
      creating the resource, but not actually perform the action.
  """
    tagKey = _messages.MessageField('TagKey', 1)
    validateOnly = _messages.BooleanField(2)