from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetQueryResultsResponse(_messages.Message):
    """A GetQueryResultsResponse object.

  Fields:
    cacheHit: Whether the query result was fetched from the query cache.
    errors: [Output-only] All errors and warnings encountered during the
      running of the job. Errors here do not necessarily mean that the job has
      completed or was unsuccessful.
    etag: A hash of this response.
    jobComplete: Whether the query has completed or not. If rows or totalRows
      are present, this will always be true. If this is false, totalRows will
      not be available.
    jobReference: Reference to the BigQuery Job that was created to run the
      query. This field will be present even if the original request timed
      out, in which case GetQueryResults can be used to read the results once
      the query has completed. Since this API only returns the first page of
      results, subsequent pages can be fetched via the same mechanism
      (GetQueryResults).
    kind: The resource type of the response.
    numDmlAffectedRows: [Output-only, Experimental] The number of rows
      affected by a DML statement. Present only for DML statements INSERT,
      UPDATE or DELETE.
    pageToken: A token used for paging results.
    rows: An object with as many results as can be contained within the
      maximum permitted reply size. To get any additional rows, you can call
      GetQueryResults and specify the jobReference returned above. Present
      only when the query completes successfully.
    schema: The schema of the results. Present only when the query completes
      successfully.
    totalBytesProcessed: The total number of bytes processed for this query.
    totalRows: The total number of rows in the complete query result set,
      which can be more than the number of rows in this single page of
      results. Present only when the query completes successfully.
  """
    cacheHit = _messages.BooleanField(1)
    errors = _messages.MessageField('ErrorProto', 2, repeated=True)
    etag = _messages.StringField(3)
    jobComplete = _messages.BooleanField(4)
    jobReference = _messages.MessageField('JobReference', 5)
    kind = _messages.StringField(6, default=u'bigquery#getQueryResultsResponse')
    numDmlAffectedRows = _messages.IntegerField(7)
    pageToken = _messages.StringField(8)
    rows = _messages.MessageField('TableRow', 9, repeated=True)
    schema = _messages.MessageField('TableSchema', 10)
    totalBytesProcessed = _messages.IntegerField(11)
    totalRows = _messages.IntegerField(12, variant=_messages.Variant.UINT64)