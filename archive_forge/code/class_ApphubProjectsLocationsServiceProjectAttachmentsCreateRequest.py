from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApphubProjectsLocationsServiceProjectAttachmentsCreateRequest(_messages.Message):
    """A ApphubProjectsLocationsServiceProjectAttachmentsCreateRequest object.

  Fields:
    parent: Required. Host project ID and location to which service project is
      being attached. Only global location is supported. Expected format:
      `projects/{project}/locations/{location}`.
    requestId: Optional. An optional request ID to identify requests. Specify
      a unique request ID so that if you must retry your request, the server
      will know to ignore the request if it has already been completed. The
      server will guarantee that for at least 60 minutes since the first
      request. For example, consider a situation where you make an initial
      request and the request times out. If you make the request again with
      the same request ID, the server can check if original operation with the
      same request ID was received, and if so, will ignore the second request.
      This prevents clients from accidentally creating duplicate commitments.
      The request ID must be a valid UUID with the exception that zero UUID is
      not supported (00000000-0000-0000-0000-000000000000).
    serviceProjectAttachment: A ServiceProjectAttachment resource to be passed
      as the request body.
    serviceProjectAttachmentId: Required. The service project attachment
      identifier must contain the project id of the service project specified
      in the service_project_attachment.service_project field.
  """
    parent = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)
    serviceProjectAttachment = _messages.MessageField('ServiceProjectAttachment', 3)
    serviceProjectAttachmentId = _messages.StringField(4)