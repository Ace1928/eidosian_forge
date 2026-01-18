from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1FeatureNoiseSigmaNoiseSigmaForFeature(_messages.Message):
    """Noise sigma for a single feature.

  Fields:
    name: The name of the input feature for which noise sigma is provided. The
      features are defined in explanation metadata inputs.
    sigma: This represents the standard deviation of the Gaussian kernel that
      will be used to add noise to the feature prior to computing gradients.
      Similar to noise_sigma but represents the noise added to the current
      feature. Defaults to 0.1.
  """
    name = _messages.StringField(1)
    sigma = _messages.FloatField(2, variant=_messages.Variant.FLOAT)