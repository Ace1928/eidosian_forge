from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudfunctionsProjectsLocationsFunctionsListRequest(_messages.Message):
    """A CloudfunctionsProjectsLocationsFunctionsListRequest object.

  Fields:
    filter: The filter for Functions that match the filter expression,
      following the syntax outlined in https://google.aip.dev/160.
    orderBy: The sorting order of the resources returned. Value should be a
      comma separated list of fields. The default sorting oder is ascending.
      See https://google.aip.dev/132#ordering.
    pageSize: Maximum number of functions to return per call. The largest
      allowed page_size is 1,000, if the page_size is omitted or specified as
      greater than 1,000 then it will be replaced as 1,000. The size of the
      list response can be less than specified when used with filters.
    pageToken: The value returned by the last `ListFunctionsResponse`;
      indicates that this is a continuation of a prior `ListFunctions` call,
      and that the system should return the next page of data.
    parent: Required. The project and location from which the function should
      be listed, specified in the format `projects/*/locations/*` If you want
      to list functions in all locations, use "-" in place of a location. When
      listing functions in all locations, if one or more location(s) are
      unreachable, the response will contain functions from all reachable
      locations along with the names of any unreachable locations.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)