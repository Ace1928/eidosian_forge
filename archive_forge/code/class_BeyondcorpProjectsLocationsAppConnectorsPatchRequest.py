from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpProjectsLocationsAppConnectorsPatchRequest(_messages.Message):
    """A BeyondcorpProjectsLocationsAppConnectorsPatchRequest object.

  Fields:
    googleCloudBeyondcorpAppconnectorsV1alphaAppConnector: A
      GoogleCloudBeyondcorpAppconnectorsV1alphaAppConnector resource to be
      passed as the request body.
    name: Required. Unique resource name of the AppConnector. The name is
      ignored when creating a AppConnector.
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
    updateMask: Required. Mask of fields to update. At least one path must be
      supplied in this field. The elements of the repeated paths field may
      only include these fields from [BeyondCorp.AppConnector]: * `labels` *
      `display_name`
    validateOnly: Optional. If set, validates request by executing a dry-run
      which would not alter the resource in any way.
  """
    googleCloudBeyondcorpAppconnectorsV1alphaAppConnector = _messages.MessageField('GoogleCloudBeyondcorpAppconnectorsV1alphaAppConnector', 1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
    updateMask = _messages.StringField(4)
    validateOnly = _messages.BooleanField(5)