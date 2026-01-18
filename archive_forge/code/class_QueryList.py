from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryList(_messages.Message):
    """A list of queries to run on a cluster.

  Fields:
    queries: Required. The queries to execute. You do not need to end a query
      expression with a semicolon. Multiple queries can be specified in one
      string by separating each with a semicolon. Here is an example of a
      Dataproc API snippet that uses a QueryList to specify a HiveJob:
      "hiveJob": { "queryList": { "queries": [ "query1", "query2",
      "query3;query4", ] } }
  """
    queries = _messages.StringField(1, repeated=True)