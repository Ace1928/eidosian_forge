from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuresourcemanagerProjectsLocationsInstancesIssueRedirectTicketInternalRequest(_messages.Message):
    """A SecuresourcemanagerProjectsLocationsInstancesIssueRedirectTicketIntern
  alRequest object.

  Fields:
    instance: Required. The instance resource to issue a redirect ticket for.
    issueRedirectTicketInternalRequest: A IssueRedirectTicketInternalRequest
      resource to be passed as the request body.
  """
    instance = _messages.StringField(1, required=True)
    issueRedirectTicketInternalRequest = _messages.MessageField('IssueRedirectTicketInternalRequest', 2)