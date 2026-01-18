from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RedisProjectsLocationsClustersListRequest(_messages.Message):
    """A RedisProjectsLocationsClustersListRequest object.

  Fields:
    pageSize: The maximum number of items to return. If not specified, a
      default value of 1000 will be used by the service. Regardless of the
      page_size value, the response may include a partial list and a caller
      should only rely on response's `next_page_token` to determine if there
      are more clusters left to be queried.
    pageToken: The `next_page_token` value returned from a previous
      ListClusters request, if any.
    parent: Required. The resource name of the cluster location using the
      form: `projects/{project_id}/locations/{location_id}` where
      `location_id` refers to a GCP region.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)