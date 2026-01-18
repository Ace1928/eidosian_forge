from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerTagBindingsCreateRequest(_messages.Message):
    """A CloudresourcemanagerTagBindingsCreateRequest object.

  Fields:
    tagBinding: A TagBinding resource to be passed as the request body.
    validateOnly: Optional. Set to true to perform the validations necessary
      for creating the resource, but not actually perform the action.
  """
    tagBinding = _messages.MessageField('TagBinding', 1)
    validateOnly = _messages.BooleanField(2)