from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1AsyncQuery(_messages.Message):
    """A GoogleCloudApigeeV1AsyncQuery object.

  Fields:
    created: Creation time of the query.
    envgroupHostname: Hostname is available only when query is executed at
      host level.
    error: Error is set when query fails.
    executionTime: ExecutionTime is available only after the query is
      completed.
    name: Asynchronous Query Name.
    queryParams: Contains information like metrics, dimenstions etc of the
      AsyncQuery.
    reportDefinitionId: Asynchronous Report ID.
    result: Result is available only after the query is completed.
    resultFileSize: ResultFileSize is available only after the query is
      completed.
    resultRows: ResultRows is available only after the query is completed.
    self: Self link of the query. Example: `/organizations/myorg/environments/
      myenv/queries/9cfc0d85-0f30-46d6-ae6f-318d0cb961bd` or following format
      if query is running at host level:
      `/organizations/myorg/hostQueries/9cfc0d85-0f30-46d6-ae6f-318d0cb961bd`
    state: Query state could be "enqueued", "running", "completed", "failed".
    updated: Last updated timestamp for the query.
  """
    created = _messages.StringField(1)
    envgroupHostname = _messages.StringField(2)
    error = _messages.StringField(3)
    executionTime = _messages.StringField(4)
    name = _messages.StringField(5)
    queryParams = _messages.MessageField('GoogleCloudApigeeV1QueryMetadata', 6)
    reportDefinitionId = _messages.StringField(7)
    result = _messages.MessageField('GoogleCloudApigeeV1AsyncQueryResult', 8)
    resultFileSize = _messages.StringField(9)
    resultRows = _messages.IntegerField(10)
    self = _messages.StringField(11)
    state = _messages.StringField(12)
    updated = _messages.StringField(13)