from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatapipelinesProjectsLocationsPipelinesListRequest(_messages.Message):
    """A DatapipelinesProjectsLocationsPipelinesListRequest object.

  Fields:
    filter: An expression for filtering the results of the request. If
      unspecified, all pipelines will be returned. Multiple filters can be
      applied and must be comma separated. Fields eligible for filtering are:
      + `type`: The type of the pipeline (streaming or batch). Allowed values
      are `ALL`, `BATCH`, and `STREAMING`. + `status`: The activity status of
      the pipeline. Allowed values are `ALL`, `ACTIVE`, `ARCHIVED`, and
      `PAUSED`. For example, to limit results to active batch processing
      pipelines: type:BATCH,status:ACTIVE
    pageSize: The maximum number of entities to return. The service may return
      fewer than this value, even if there are additional pages. If
      unspecified, the max limit is yet to be determined by the backend
      implementation.
    pageToken: A page token, received from a previous `ListPipelines` call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to `ListPipelines` must match the call that provided
      the page token.
    parent: Required. The location name. For example:
      `projects/PROJECT_ID/locations/LOCATION_ID`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)