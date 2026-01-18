from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunQueryResponse(_messages.Message):
    """The response for Firestore.RunQuery.

  Fields:
    document: A query result, not set when reporting partial progress.
    done: If present, Firestore has completely finished the request and no
      more documents will be returned.
    explainMetrics: Query explain metrics. This is only present when the
      RunQueryRequest.explain_options is provided, and it is sent only once
      with the last response in the stream.
    readTime: The time at which the document was read. This may be
      monotonically increasing; in this case, the previous documents in the
      result stream are guaranteed not to have changed between their
      `read_time` and this one. If the query returns no results, a response
      with `read_time` and no `document` will be sent, and this represents the
      time at which the query was run.
    skippedResults: The number of results that have been skipped due to an
      offset between the last response and the current response.
    transaction: The transaction that was started as part of this request. Can
      only be set in the first response, and only if
      RunQueryRequest.new_transaction was set in the request. If set, no other
      fields will be set in this response.
  """
    document = _messages.MessageField('Document', 1)
    done = _messages.BooleanField(2)
    explainMetrics = _messages.MessageField('ExplainMetrics', 3)
    readTime = _messages.StringField(4)
    skippedResults = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    transaction = _messages.BytesField(6)