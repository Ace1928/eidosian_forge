from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2ExternalSystem(_messages.Message):
    """Representation of third party SIEM/SOAR fields within SCC.

  Fields:
    assignees: References primary/secondary etc assignees in the external
      system.
    caseCloseTime: The time when the case was closed, as reported by the
      external system.
    caseCreateTime: The time when the case was created, as reported by the
      external system.
    casePriority: The priority of the finding's corresponding case in the
      external system.
    caseSla: The SLA of the finding's corresponding case in the external
      system.
    caseUri: The link to the finding's corresponding case in the external
      system.
    externalSystemUpdateTime: The time when the case was last updated, as
      reported by the external system.
    externalUid: The identifier that's used to track the finding's
      corresponding case in the external system.
    name: Full resource name of the external system. The following list shows
      some examples: +
      `organizations/1234/sources/5678/findings/123456/externalSystems/jira` +
      `organizations/1234/sources/5678/locations/us/findings/123456/externalSy
      stems/jira` +
      `folders/1234/sources/5678/findings/123456/externalSystems/jira` + `fold
      ers/1234/sources/5678/locations/us/findings/123456/externalSystems/jira`
      + `projects/1234/sources/5678/findings/123456/externalSystems/jira` + `p
      rojects/1234/sources/5678/locations/us/findings/123456/externalSystems/j
      ira`
    status: The most recent status of the finding's corresponding case, as
      reported by the external system.
    ticketInfo: Information about the ticket, if any, that is being used to
      track the resolution of the issue that is identified by this finding.
  """
    assignees = _messages.StringField(1, repeated=True)
    caseCloseTime = _messages.StringField(2)
    caseCreateTime = _messages.StringField(3)
    casePriority = _messages.StringField(4)
    caseSla = _messages.StringField(5)
    caseUri = _messages.StringField(6)
    externalSystemUpdateTime = _messages.StringField(7)
    externalUid = _messages.StringField(8)
    name = _messages.StringField(9)
    status = _messages.StringField(10)
    ticketInfo = _messages.MessageField('GoogleCloudSecuritycenterV2TicketInfo', 11)