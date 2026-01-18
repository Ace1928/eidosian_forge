from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1FeatureNoiseSigma(_messages.Message):
    """Noise sigma by features. Noise sigma represents the standard deviation
  of the gaussian kernel that will be used to add noise to interpolated inputs
  prior to computing gradients.

  Fields:
    noiseSigma: Noise sigma per feature. No noise is added to features that
      are not set.
  """
    noiseSigma = _messages.MessageField('GoogleCloudAiplatformV1FeatureNoiseSigmaNoiseSigmaForFeature', 1, repeated=True)