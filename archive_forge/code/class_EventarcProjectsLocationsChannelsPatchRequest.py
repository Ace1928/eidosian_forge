from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventarcProjectsLocationsChannelsPatchRequest(_messages.Message):
    """A EventarcProjectsLocationsChannelsPatchRequest object.

  Fields:
    channel: A Channel resource to be passed as the request body.
    name: Required. The resource name of the channel. Must be unique within
      the location on the project and must be in
      `projects/{project}/locations/{location}/channels/{channel_id}` format.
    updateMask: The fields to be updated; only fields explicitly provided are
      updated. If no field mask is provided, all provided fields in the
      request are updated. To update all fields, provide a field mask of "*".
    validateOnly: Optional. If set, validate the request and preview the
      review, but do not post it.
  """
    channel = _messages.MessageField('Channel', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)