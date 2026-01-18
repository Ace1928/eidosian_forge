from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupdrProjectsLocationsManagementServersCreateRequest(_messages.Message):
    """A BackupdrProjectsLocationsManagementServersCreateRequest object.

  Fields:
    managementServer: A ManagementServer resource to be passed as the request
      body.
    managementServerId: Required. The name of the management server to create.
      The name must be unique for the specified project and location.
    parent: Required. The management server project and location in the format
      `projects/{project_id}/locations/{location}`. In Cloud Backup and DR
      locations map to GCP regions, for example **us-central1**.
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
  """
    managementServer = _messages.MessageField('ManagementServer', 1)
    managementServerId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)