from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1StatsEnvironmentStats(_messages.Message):
    """Encapsulates the environment wrapper: ``` "environments": [ { "metrics":
  [ { "name": "sum(message_count)", "values": [ "2.52056245E8" ] } ], "name":
  "prod" } ]```

  Fields:
    dimensions: List of metrics grouped under dimensions.
    metrics: In the final response, only one of the following fields will be
      present based on the dimensions provided. If no dimensions are provided,
      then only top-level metrics is provided. If dimensions are included,
      then there will be a top-level dimensions field under environments which
      will contain metrics values and the dimension name. Example: ```
      "environments": [ { "dimensions": [ { "metrics": [ { "name":
      "sum(message_count)", "values": [ "2.14049521E8" ] } ], "name":
      "nit_proxy" } ], "name": "prod" } ]``` or ```"environments": [ {
      "metrics": [ { "name": "sum(message_count)", "values": [ "2.19026331E8"
      ] } ], "name": "prod" } ]``` List of metric values.
    name: Name of the environment.
  """
    dimensions = _messages.MessageField('GoogleCloudApigeeV1DimensionMetric', 1, repeated=True)
    metrics = _messages.MessageField('GoogleCloudApigeeV1Metric', 2, repeated=True)
    name = _messages.StringField(3)