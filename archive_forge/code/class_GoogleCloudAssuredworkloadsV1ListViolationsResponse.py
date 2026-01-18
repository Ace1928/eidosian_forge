from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1ListViolationsResponse(_messages.Message):
    """Response of ListViolations endpoint.

  Fields:
    nextPageToken: The next page token. Returns empty if reached the last
      page.
    violations: List of Violations under a Workload.
  """
    nextPageToken = _messages.StringField(1)
    violations = _messages.MessageField('GoogleCloudAssuredworkloadsV1Violation', 2, repeated=True)