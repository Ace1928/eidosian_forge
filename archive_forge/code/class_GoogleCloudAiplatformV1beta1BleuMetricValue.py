from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BleuMetricValue(_messages.Message):
    """Bleu metric value for an instance.

  Fields:
    score: Output only. Bleu score.
  """
    score = _messages.FloatField(1, variant=_messages.Variant.FLOAT)