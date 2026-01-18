from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkebackupProjectsLocationsBackupPlansListRequest(_messages.Message):
    """A GkebackupProjectsLocationsBackupPlansListRequest object.

  Fields:
    filter: Optional. Field match expression used to filter the results.
    orderBy: Optional. Field by which to sort the results.
    pageSize: Optional. The target number of results to return in a single
      response. If not specified, a default value will be chosen by the
      service. Note that the response may include a partial list and a caller
      should only rely on the response's next_page_token to determine if there
      are more instances left to be queried.
    pageToken: Optional. The value of next_page_token received from a previous
      `ListBackupPlans` call. Provide this to retrieve the subsequent page in
      a multi-page list of results. When paginating, all other parameters
      provided to `ListBackupPlans` must match the call that provided the page
      token.
    parent: Required. The location that contains the BackupPlans to list.
      Format: `projects/*/locations/*`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)