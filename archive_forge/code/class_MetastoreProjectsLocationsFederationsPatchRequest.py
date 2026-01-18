from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetastoreProjectsLocationsFederationsPatchRequest(_messages.Message):
    """A MetastoreProjectsLocationsFederationsPatchRequest object.

  Fields:
    federation: A Federation resource to be passed as the request body.
    name: Immutable. The relative resource name of the federation, of the
      form: projects/{project_number}/locations/{location_id}/federations/{fed
      eration_id}`.
    requestId: Optional. A request ID. Specify a unique request ID to allow
      the server to ignore the request if it has completed. The server will
      ignore subsequent requests that provide a duplicate request ID for at
      least 60 minutes after the first request.For example, if an initial
      request times out, followed by another request with the same request ID,
      the server ignores the second request to prevent the creation of
      duplicate commitments.The request ID must be a valid UUID
      (https://en.wikipedia.org/wiki/Universally_unique_identifier#Format) A
      zero UUID (00000000-0000-0000-0000-000000000000) is not supported.
    updateMask: Required. A field mask used to specify the fields to be
      overwritten in the metastore federation resource by the update. Fields
      specified in the update_mask are relative to the resource (not to the
      full request). A field is overwritten if it is in the mask.
  """
    federation = _messages.MessageField('Federation', 1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
    updateMask = _messages.StringField(4)