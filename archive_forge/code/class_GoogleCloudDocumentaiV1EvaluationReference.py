from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1EvaluationReference(_messages.Message):
    """Gives a short summary of an evaluation, and links to the evaluation
  itself.

  Fields:
    aggregateMetrics: An aggregate of the statistics for the evaluation with
      fuzzy matching on.
    aggregateMetricsExact: An aggregate of the statistics for the evaluation
      with fuzzy matching off.
    evaluation: The resource name of the evaluation.
    operation: The resource name of the Long Running Operation for the
      evaluation.
  """
    aggregateMetrics = _messages.MessageField('GoogleCloudDocumentaiV1EvaluationMetrics', 1)
    aggregateMetricsExact = _messages.MessageField('GoogleCloudDocumentaiV1EvaluationMetrics', 2)
    evaluation = _messages.StringField(3)
    operation = _messages.StringField(4)