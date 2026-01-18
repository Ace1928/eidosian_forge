from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatapipelinesProjectsLocationsPipelinesJobsListRequest(_messages.Message):
    """A DatapipelinesProjectsLocationsPipelinesJobsListRequest object.

  Fields:
    pageSize: The maximum number of entities to return. The service may return
      fewer than this value, even if there are additional pages. If
      unspecified, the max limit will be determined by the backend
      implementation.
    pageToken: A page token, received from a previous `ListJobs` call. Provide
      this to retrieve the subsequent page. When paginating, all other
      parameters provided to `ListJobs` must match the call that provided the
      page token.
    parent: Required. The pipeline name. For example:
      `projects/PROJECT_ID/locations/LOCATION_ID/pipelines/PIPELINE_ID`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)