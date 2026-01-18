from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesClustersHotTabletsListRequest(_messages.Message):
    """A BigtableadminProjectsInstancesClustersHotTabletsListRequest object.

  Fields:
    endTime: The end time to list hot tablets.
    pageSize: Maximum number of results per page. A page_size that is empty or
      zero lets the server choose the number of items to return. A page_size
      which is strictly positive will return at most that many items. A
      negative page_size will cause an error. Following the first request,
      subsequent paginated calls do not need a page_size field. If a page_size
      is set in subsequent calls, it must match the page_size given in the
      first request.
    pageToken: The value of `next_page_token` returned by a previous call.
    parent: Required. The cluster name to list hot tablets. Value is in the
      following form:
      `projects/{project}/instances/{instance}/clusters/{cluster}`.
    startTime: The start time to list hot tablets. The hot tablets in the
      response will have start times between the requested start time and end
      time. Start time defaults to Now if it is unset, and end time defaults
      to Now - 24 hours if it is unset. The start time should be less than the
      end time, and the maximum allowed time range between start time and end
      time is 48 hours. Start time and end time should have values between Now
      and Now - 14 days.
  """
    endTime = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    startTime = _messages.StringField(5)