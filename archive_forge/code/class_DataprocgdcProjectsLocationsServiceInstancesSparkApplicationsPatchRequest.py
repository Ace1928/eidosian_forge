from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocgdcProjectsLocationsServiceInstancesSparkApplicationsPatchRequest(_messages.Message):
    """A
  DataprocgdcProjectsLocationsServiceInstancesSparkApplicationsPatchRequest
  object.

  Fields:
    allowMissing: Optional. Whether to upsert this application if it does not
      exist already In this situation, `update_mask` is ignored.
    name: Identifier. Fields 1-6 should exist for all declarative friendly
      resources per aip.dev/148 The name of the application. Format: projects/
      {project}/locations/{location}/serviceInstances/{service_instance}/spark
      Applications/{application}
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
    updateMask: Required. The list of fields to update.
  """
    allowMissing = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
    sparkApplication = _messages.MessageField('SparkApplication', 4)
    updateMask = _messages.StringField(5)