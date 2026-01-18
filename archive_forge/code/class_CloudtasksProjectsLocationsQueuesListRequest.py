from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudtasksProjectsLocationsQueuesListRequest(_messages.Message):
    """A CloudtasksProjectsLocationsQueuesListRequest object.

  Fields:
    filter: `filter` can be used to specify a subset of queues. Any Queue
      field can be used as a filter and several operators as supported. For
      example: `<=, <, >=, >, !=, =, :`. The filter syntax is the same as
      described in [Stackdriver's Advanced Logs
      Filters](https://cloud.google.com/logging/docs/view/advanced_filters).
      Sample filter "state: PAUSED". Note that using filters might cause fewer
      queues than the requested page_size to be returned.
    pageSize: Requested page size. The maximum page size is 9800. If
      unspecified, the page size will be the maximum. Fewer queues than
      requested might be returned, even if more queues exist; use the
      next_page_token in the response to determine if more queues exist.
    pageToken: A token identifying the page of results to return. To request
      the first page results, page_token must be empty. To request the next
      page of results, page_token must be the value of next_page_token
      returned from the previous call to ListQueues method. It is an error to
      switch the value of the filter while iterating through pages.
    parent: Required. The location name. For example:
      `projects/PROJECT_ID/locations/LOCATION_ID`
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)