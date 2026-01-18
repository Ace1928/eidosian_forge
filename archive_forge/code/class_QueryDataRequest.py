from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryDataRequest(_messages.Message):
    """The parameters to QueryData.

  Fields:
    disableQueryCaching: Optional. If set to true, turns off all query caching
      on both the Log Analytics and BigQuery sides.
    query: Optional. The contents of the query. If this field is populated,
      query_steps will be ignored.
    resourceNames: Required. Names of one or more log views to run a SQL
      query.Example: projects/[PROJECT_ID]/locations/[LOCATION_ID]/buckets/[BU
      CKET_ID]/views/[VIEW_ID]Requires appropriate permissions on each
      resource such as 'logging.views.access' on log view resources.
    timeout: Optional. The timeout for the query. BigQuery will terminate the
      job if this duration is exceeded. If not set, the default is 5 minutes.
  """
    disableQueryCaching = _messages.BooleanField(1)
    query = _messages.MessageField('AnalyticsQuery', 2)
    resourceNames = _messages.StringField(3, repeated=True)
    timeout = _messages.StringField(4)