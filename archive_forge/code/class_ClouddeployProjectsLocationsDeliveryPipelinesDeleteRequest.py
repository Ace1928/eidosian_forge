from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsDeliveryPipelinesDeleteRequest(_messages.Message):
    """A ClouddeployProjectsLocationsDeliveryPipelinesDeleteRequest object.

  Fields:
    allowMissing: Optional. If set to true, then deleting an already deleted
      or non-existing `DeliveryPipeline` will succeed.
    etag: Optional. This checksum is computed by the server based on the value
      of other fields, and may be sent on update and delete requests to ensure
      the client has an up-to-date value before proceeding.
    force: Optional. If set to true, all child resources under this pipeline
      will also be deleted. Otherwise, the request will only work if the
      pipeline has no child resources.
    name: Required. The name of the `DeliveryPipeline` to delete. Format
      should be `projects/{project_id}/locations/{location_name}/deliveryPipel
      ines/{pipeline_name}`.
    requestId: Optional. A request ID to identify requests. Specify a unique
      request ID so that if you must retry your request, the server knows to
      ignore the request if it has already been completed. The server
      guarantees that for at least 60 minutes after the first request. For
      example, consider a situation where you make an initial request and the
      request times out. If you make the request again with the same request
      ID, the server can check if original operation with the same request ID
      was received, and if so, will ignore the second request. This prevents
      clients from accidentally creating duplicate commitments. The request ID
      must be a valid UUID with the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
    validateOnly: Optional. If set, validate the request and preview the
      review, but do not actually post it.
  """
    allowMissing = _messages.BooleanField(1)
    etag = _messages.StringField(2)
    force = _messages.BooleanField(3)
    name = _messages.StringField(4, required=True)
    requestId = _messages.StringField(5)
    validateOnly = _messages.BooleanField(6)