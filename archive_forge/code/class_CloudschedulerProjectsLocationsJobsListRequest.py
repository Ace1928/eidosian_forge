from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudschedulerProjectsLocationsJobsListRequest(_messages.Message):
    """A CloudschedulerProjectsLocationsJobsListRequest object.

  Fields:
    pageSize: Requested page size.  The maximum page size is 500. If
      unspecified, the page size will be the maximum. Fewer jobs than
      requested might be returned, even if more jobs exist; use
      next_page_token to determine if more jobs exist.
    pageToken: A token identifying a page of results the server will return.
      To request the first page results, page_token must be empty. To request
      the next page of results, page_token must be the value of
      next_page_token returned from the previous call to ListJobs. It is an
      error to switch the value of filter or order_by while iterating through
      pages.
    parent: Required.  The location name. For example:
      `projects/PROJECT_ID/locations/LOCATION_ID`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)