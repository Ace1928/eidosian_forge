from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BleuSpec(_messages.Message):
    """Spec for bleu score metric - calculates the precision of n-grams in the
  prediction as compared to reference - returns a score ranging between 0 to
  1.
  """