from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetastoreProjectsLocationsServicesPatchRequest(_messages.Message):
    """A MetastoreProjectsLocationsServicesPatchRequest object.

  Fields:
    name: Immutable. The relative resource name of the metastore service, in
      the following format:projects/{project_number}/locations/{location_id}/s
      ervices/{service_id}.
    requestId: Optional. A request ID. Specify a unique request ID to allow
      the server to ignore the request if it has completed. The server will
      ignore subsequent requests that provide a duplicate request ID for at
      least 60 minutes after the first request.For example, if an initial
      request times out, followed by another request with the same request ID,
      the server ignores the second request to prevent the creation of
      duplicate commitments.The request ID must be a valid UUID
      (https://en.wikipedia.org/wiki/Universally_unique_identifier#Format) A
      zero UUID (00000000-0000-0000-0000-000000000000) is not supported.
    service: A Service resource to be passed as the request body.
    updateMask: Required. A field mask used to specify the fields to be
      overwritten in the metastore service resource by the update. Fields
      specified in the update_mask are relative to the resource (not to the
      full request). A field is overwritten if it is in the mask.
  """
    name = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)
    service = _messages.MessageField('Service', 3)
    updateMask = _messages.StringField(4)