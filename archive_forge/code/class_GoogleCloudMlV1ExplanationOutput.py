from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1ExplanationOutput(_messages.Message):
    """Represents results of an explanation job.

  Fields:
    errorCount: The number of data instances which resulted in errors.
    explanationCount: The number of generated explanations.
    nodeHours: Node hours used by the batch explanation job.
    outputBigqueryTable: The output BigQuery table name provided at the job
      creation time.
  """
    errorCount = _messages.IntegerField(1)
    explanationCount = _messages.IntegerField(2)
    nodeHours = _messages.FloatField(3)
    outputBigqueryTable = _messages.StringField(4)