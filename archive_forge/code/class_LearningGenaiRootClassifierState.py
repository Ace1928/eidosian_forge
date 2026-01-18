from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootClassifierState(_messages.Message):
    """DataProviderOutput and MetricOutput can be saved between calls to the
  Classifier framework. For instance, you can run the query classifier, get
  outputs from those metrics, then use them in a result classifier as well.
  Example rule based on this idea: and_rules { rule { metric_name:
  'query_safesearch_v2' ... } rule { metric_name: 'response_safesearch_v2' ...
  } }

  Fields:
    dataProviderOutput: A LearningGenaiRootDataProviderOutput attribute.
    metricOutput: A LearningGenaiRootMetricOutput attribute.
  """
    dataProviderOutput = _messages.MessageField('LearningGenaiRootDataProviderOutput', 1, repeated=True)
    metricOutput = _messages.MessageField('LearningGenaiRootMetricOutput', 2, repeated=True)