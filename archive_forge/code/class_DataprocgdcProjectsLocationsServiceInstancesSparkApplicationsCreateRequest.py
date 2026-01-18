from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocgdcProjectsLocationsServiceInstancesSparkApplicationsCreateRequest(_messages.Message):
    """A
  DataprocgdcProjectsLocationsServiceInstancesSparkApplicationsCreateRequest
  object.

  Fields:
    parent: Required. The parent location resource where this application will
      be created
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
    sparkApplication: A SparkApplication resource to be passed as the request
      body.
    sparkApplicationId: Required. The id of the application
  """
    parent = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)
    sparkApplication = _messages.MessageField('SparkApplication', 3)
    sparkApplicationId = _messages.StringField(4)