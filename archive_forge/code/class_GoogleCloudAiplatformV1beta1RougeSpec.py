from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1RougeSpec(_messages.Message):
    """Spec for rouge score metric - calculates the recall of n-grams in
  prediction as compared to reference - returns a score ranging between 0 and
  1.

  Fields:
    rougeType: Optional. Supported rouge types are rougen[1-9], rougeL and
      rougeLsum.
    splitSummaries: Optional. Whether to split summaries while using
      rougeLsum.
    useStemmer: Optional. Whether to use stemmer to compute rouge score.
  """
    rougeType = _messages.StringField(1)
    splitSummaries = _messages.BooleanField(2)
    useStemmer = _messages.BooleanField(3)