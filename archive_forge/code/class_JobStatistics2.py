from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobStatistics2(_messages.Message):
    """A JobStatistics2 object.

  Fields:
    billingTier: [Output-only] Billing tier for the job.
    cacheHit: [Output-only] Whether the query result was fetched from the
      query cache.
    numDmlAffectedRows: [Output-only, Experimental] The number of rows
      affected by a DML statement. Present only for DML statements INSERT,
      UPDATE or DELETE.
    queryPlan: [Output-only, Experimental] Describes execution plan for the
      query.
    referencedTables: [Output-only, Experimental] Referenced tables for the
      job. Queries that reference more than 50 tables will not have a complete
      list.
    schema: [Output-only, Experimental] The schema of the results. Present
      only for successful dry run of non-legacy SQL queries.
    totalBytesBilled: [Output-only] Total bytes billed for the job.
    totalBytesProcessed: [Output-only] Total bytes processed for the job.
  """
    billingTier = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    cacheHit = _messages.BooleanField(2)
    numDmlAffectedRows = _messages.IntegerField(3)
    queryPlan = _messages.MessageField('ExplainQueryStage', 4, repeated=True)
    referencedTables = _messages.MessageField('TableReference', 5, repeated=True)
    schema = _messages.MessageField('TableSchema', 6)
    totalBytesBilled = _messages.IntegerField(7)
    totalBytesProcessed = _messages.IntegerField(8)