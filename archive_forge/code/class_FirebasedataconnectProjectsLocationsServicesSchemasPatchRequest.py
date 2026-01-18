from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirebasedataconnectProjectsLocationsServicesSchemasPatchRequest(_messages.Message):
    """A FirebasedataconnectProjectsLocationsServicesSchemasPatchRequest
  object.

  Fields:
    allowMissing: Optional. If true and the Schema is not found, a new Schema
      will be created. In this case, `update_mask` is ignored.
    name: Identifier. The relative resource name of the schema, in the format:
      ``` projects/{project}/locations/{location}/services/{service}/schemas/{
      schema} ``` Right now, the only supported schema is "main".
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
    schema: A Schema resource to be passed as the request body.
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the Schema resource by the update. The fields specified
      in the update_mask are relative to the resource, not the full request. A
      field will be overwritten if it is in the mask. If the user does not
      provide a mask then all fields will be overwritten.
    validateOnly: Optional. If set, validate the request and preview the
      Schema, but do not actually update it.
  """
    allowMissing = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
    schema = _messages.MessageField('Schema', 4)
    updateMask = _messages.StringField(5)
    validateOnly = _messages.BooleanField(6)