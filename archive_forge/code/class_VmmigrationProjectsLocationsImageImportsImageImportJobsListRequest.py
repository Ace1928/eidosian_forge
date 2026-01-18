from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmmigrationProjectsLocationsImageImportsImageImportJobsListRequest(_messages.Message):
    """A VmmigrationProjectsLocationsImageImportsImageImportJobsListRequest
  object.

  Fields:
    filter: Optional. The filter request (according to
      https://google.aip.dev/160).
    orderBy: Optional. The order by fields for the result (according to
      https://google.aip.dev/132#ordering). Currently ordering is only
      possible by "name" field.
    pageSize: Optional. The maximum number of targets to return. The service
      may return fewer than this value. If unspecified, at most 500 targets
      will be returned. The maximum value is 1000; values above 1000 will be
      coerced to 1000.
    pageToken: Optional. A page token, received from a previous
      `ListImageImportJobs` call. Provide this to retrieve the subsequent
      page. When paginating, all other parameters provided to
      `ListImageImportJobs` must match the call that provided the page token.
    parent: Required. The parent, which owns this collection of targets.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)