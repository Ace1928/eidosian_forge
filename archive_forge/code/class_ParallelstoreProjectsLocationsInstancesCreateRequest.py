from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ParallelstoreProjectsLocationsInstancesCreateRequest(_messages.Message):
    """A ParallelstoreProjectsLocationsInstancesCreateRequest object.

  Fields:
    instance: A Instance resource to be passed as the request body.
    instanceId: Required. The logical name of the Parallelstore instance in
      the user project with the following restrictions: * Must contain only
      lowercase letters, numbers, and hyphens. * Must start with a letter. *
      Must be between 1-63 characters. * Must end with a number or a letter. *
      Must be unique within the customer project / location
    parent: Required. The instance's project and location, in the format
      `projects/{project}/locations/{location}`. Locations map to Google Cloud
      zones, for example **us-west1-b**.
    requestId: Optional. An optional request ID to identify requests. Specify
      a unique request ID so that if you must retry your request, the server
      will know to ignore the request if it has already been completed. The
      server will guarantee that for at least 60 minutes since the first
      request. For example, consider a situation where you make an initial
      request and t he request times out. If you make the request again with
      the same request ID, the server can check if original operation with the
      same request ID was received, and if so, will ignore the second request.
      This prevents clients from accidentally creating duplicate commitments.
      The request ID must be a valid UUID with the exception that zero UUID is
      not supported (00000000-0000-0000-0000-000000000000).
  """
    instance = _messages.MessageField('Instance', 1)
    instanceId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)