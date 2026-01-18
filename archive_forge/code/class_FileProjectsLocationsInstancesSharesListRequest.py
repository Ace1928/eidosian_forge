from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileProjectsLocationsInstancesSharesListRequest(_messages.Message):
    """A FileProjectsLocationsInstancesSharesListRequest object.

  Fields:
    filter: List filter.
    orderBy: Sort results. Supported values are "name", "name desc" or ""
      (unsorted).
    pageSize: The maximum number of items to return.
    pageToken: The next_page_token value to use if there are additional
      results to retrieve for this list request.
    parent: Required. The instance for which to retrieve share information, in
      the format
      `projects/{project_id}/locations/{location}/instances/{instance_id}`.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)