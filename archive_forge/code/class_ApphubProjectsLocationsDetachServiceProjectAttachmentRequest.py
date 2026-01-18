from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApphubProjectsLocationsDetachServiceProjectAttachmentRequest(_messages.Message):
    """A ApphubProjectsLocationsDetachServiceProjectAttachmentRequest object.

  Fields:
    detachServiceProjectAttachmentRequest: A
      DetachServiceProjectAttachmentRequest resource to be passed as the
      request body.
    name: Required. Service project id and location to detach from a host
      project. Only global location is supported. Expected format:
      `projects/{project}/locations/{location}`.
  """
    detachServiceProjectAttachmentRequest = _messages.MessageField('DetachServiceProjectAttachmentRequest', 1)
    name = _messages.StringField(2, required=True)