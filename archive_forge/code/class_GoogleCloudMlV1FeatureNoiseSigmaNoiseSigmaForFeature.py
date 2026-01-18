from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1FeatureNoiseSigmaNoiseSigmaForFeature(_messages.Message):
    """Noise sigma for a single feature.

  Fields:
    name: The name of the input feature for which noise sigma is provided.
    sigma: Standard deviation of gaussian kernel for noise.
  """
    name = _messages.StringField(1)
    sigma = _messages.FloatField(2, variant=_messages.Variant.FLOAT)