from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListUpgradeJobsResponse(_messages.Message):
    """Response message for VmwareEngine.ListUpgradeJobs.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    unreachable: Unreachable resources.
    upgradeJobs: A list of Upgrade Jobs.
  """
    nextPageToken = _messages.StringField(1)
    unreachable = _messages.StringField(2, repeated=True)
    upgradeJobs = _messages.MessageField('UpgradeJob', 3, repeated=True)