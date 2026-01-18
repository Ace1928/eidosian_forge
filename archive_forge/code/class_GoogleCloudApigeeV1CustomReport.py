from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1CustomReport(_messages.Message):
    """A GoogleCloudApigeeV1CustomReport object.

  Fields:
    chartType: This field contains the chart type for the report
    comments: Legacy field: not used. This field contains a list of comments
      associated with custom report
    createdAt: Output only. Unix time when the app was created json key:
      createdAt
    dimensions: This contains the list of dimensions for the report
    displayName: This is the display name for the report
    environment: Output only. Environment name
    filter: This field contains the filter expression
    fromTime: Legacy field: not used. Contains the from time for the report
    lastModifiedAt: Output only. Modified time of this entity as milliseconds
      since epoch. json key: lastModifiedAt
    lastViewedAt: Output only. Last viewed time of this entity as milliseconds
      since epoch
    limit: Legacy field: not used This field contains the limit for the result
      retrieved
    metrics: Required. This contains the list of metrics
    name: Required. Unique identifier for the report T his is a legacy field
      used to encode custom report unique id
    offset: Legacy field: not used. This field contains the offset for the
      data
    organization: Output only. Organization name
    properties: This field contains report properties such as ui metadata etc.
    sortByCols: Legacy field: not used much. Contains the list of sort by
      columns
    sortOrder: Legacy field: not used much. Contains the sort order for the
      sort columns
    tags: Legacy field: not used. This field contains a list of tags
      associated with custom report
    timeUnit: This field contains the time unit of aggregation for the report
    toTime: Legacy field: not used. Contains the end time for the report
    topk: Legacy field: not used. This field contains the top k parameter
      value for restricting the result
  """
    chartType = _messages.StringField(1)
    comments = _messages.StringField(2, repeated=True)
    createdAt = _messages.IntegerField(3)
    dimensions = _messages.StringField(4, repeated=True)
    displayName = _messages.StringField(5)
    environment = _messages.StringField(6)
    filter = _messages.StringField(7)
    fromTime = _messages.StringField(8)
    lastModifiedAt = _messages.IntegerField(9)
    lastViewedAt = _messages.IntegerField(10)
    limit = _messages.StringField(11)
    metrics = _messages.MessageField('GoogleCloudApigeeV1CustomReportMetric', 12, repeated=True)
    name = _messages.StringField(13)
    offset = _messages.StringField(14)
    organization = _messages.StringField(15)
    properties = _messages.MessageField('GoogleCloudApigeeV1ReportProperty', 16, repeated=True)
    sortByCols = _messages.StringField(17, repeated=True)
    sortOrder = _messages.StringField(18)
    tags = _messages.StringField(19, repeated=True)
    timeUnit = _messages.StringField(20)
    toTime = _messages.StringField(21)
    topk = _messages.StringField(22)