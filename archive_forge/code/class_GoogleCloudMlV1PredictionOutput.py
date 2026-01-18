from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1PredictionOutput(_messages.Message):
    """Represents results of a prediction job.

  Fields:
    errorCount: The number of data instances which resulted in errors.
    nodeHours: Node hours used by the batch prediction job.
    outputPath: The output Google Cloud Storage location provided at the job
      creation time.
    predictionCount: The number of generated predictions.
  """
    errorCount = _messages.IntegerField(1)
    nodeHours = _messages.FloatField(2)
    outputPath = _messages.StringField(3)
    predictionCount = _messages.IntegerField(4)