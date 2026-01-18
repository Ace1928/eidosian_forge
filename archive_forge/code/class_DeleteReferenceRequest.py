from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeleteReferenceRequest(_messages.Message):
    """The DeleteReferenceRequest request.

  Fields:
    name: Required. Full resource name of the reference, in the following
      format:
      `//{targer_service}/{target_resource}/references/{reference_id}`. For
      example: `//targetservice.googleapis.com/projects/{my-
      project}/locations/{location}/instances/{my-instance}/references/{xyz}`.
    requestId: Optional. Request ID is an idempotency ID of the request. It
      must be a valid UUID. Zero UUID (00000000-0000-0000-0000-000000000000)
      is not supported.
  """
    name = _messages.StringField(1)
    requestId = _messages.StringField(2)