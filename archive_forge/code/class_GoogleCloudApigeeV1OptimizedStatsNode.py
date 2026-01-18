from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1OptimizedStatsNode(_messages.Message):
    """Encapsulates a data node as represented below: ``` { "identifier": {
  "names": [ "apiproxy" ], "values": [ "sirjee" ] }, "metric": [ { "env":
  "prod", "name": "sum(message_count)", "values": [ 36.0 ] } ] }``` or ``` {
  "env": "prod", "name": "sum(message_count)", "values": [ 36.0 ] }```
  Depending on whether a dimension is present in the query or not the data
  node type can be a simple metric value or dimension identifier with list of
  metrics.

  Fields:
    data: A extra_types.JsonValue attribute.
  """
    data = _messages.MessageField('extra_types.JsonValue', 1, repeated=True)