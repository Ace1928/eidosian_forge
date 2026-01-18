from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsQueriesListRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsQueriesListRequest object.

  Fields:
    dataset: Filter response list by dataset. Example: `api`, `mint`
    from_: Filter response list by returning asynchronous queries that created
      after this date time. Time must be in ISO date-time format like
      '2011-12-03T10:15:30Z'.
    inclQueriesWithoutReport: Flag to include asynchronous queries that don't
      have a report denifition.
    parent: Required. The parent resource name. Must be of the form
      `organizations/{org}/environments/{env}`.
    status: Filter response list by asynchronous query status.
    submittedBy: Filter response list by user who submitted queries.
    to: Filter response list by returning asynchronous queries that created
      before this date time. Time must be in ISO date-time format like
      '2011-12-03T10:16:30Z'.
  """
    dataset = _messages.StringField(1)
    from_ = _messages.StringField(2)
    inclQueriesWithoutReport = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    status = _messages.StringField(5)
    submittedBy = _messages.StringField(6)
    to = _messages.StringField(7)