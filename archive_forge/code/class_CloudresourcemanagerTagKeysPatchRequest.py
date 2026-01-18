from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerTagKeysPatchRequest(_messages.Message):
    """A CloudresourcemanagerTagKeysPatchRequest object.

  Fields:
    name: Immutable. The resource name for a TagKey. Must be in the format
      `tagKeys/{tag_key_id}`, where `tag_key_id` is the generated numeric id
      for the TagKey.
    tagKey: A TagKey resource to be passed as the request body.
    updateMask: Fields to be updated. The mask may only contain `description`
      or `etag`. If omitted entirely, both `description` and `etag` are
      assumed to be significant.
    validateOnly: Set as true to perform validations necessary for updating
      the resource, but not actually perform the action.
  """
    name = _messages.StringField(1, required=True)
    tagKey = _messages.MessageField('TagKey', 2)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)