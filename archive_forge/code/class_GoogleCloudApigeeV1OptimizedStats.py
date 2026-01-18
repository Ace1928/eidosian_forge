from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1OptimizedStats(_messages.Message):
    """A GoogleCloudApigeeV1OptimizedStats object.

  Fields:
    Response: Wraps the `stats` response for JavaScript Optimized Scenario
      with a response key. For example: ```{ "Response": { "TimeUnit": [],
      "metaData": { "errors": [], "notices": [ "Source:Postgres", "Table used:
      edge.api.aaxgroup001.agg_api", "PG
      Host:ruappg08-ro.production.apigeeks.net", "query served
      by:80c4ebca-6a10-4a2e-8faf-c60c1ee306ca" ] }, "resultTruncated": false,
      "stats": { "data": [ { "identifier": { "names": [ "apiproxy" ],
      "values": [ "sirjee" ] }, "metric": [ { "env": "prod", "name":
      "sum(message_count)", "values": [ 36.0 ] }, { "env": "prod", "name":
      "sum(is_error)", "values": [ 36.0 ] } ] } ] } } }```
  """
    Response = _messages.MessageField('GoogleCloudApigeeV1OptimizedStatsResponse', 1)