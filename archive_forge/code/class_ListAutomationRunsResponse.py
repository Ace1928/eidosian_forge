from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAutomationRunsResponse(_messages.Message):
    """The response object from `ListAutomationRuns`.

  Fields:
    automationRuns: The `AutomationRuns` objects.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    unreachable: Locations that could not be reached.
  """
    automationRuns = _messages.MessageField('AutomationRun', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)