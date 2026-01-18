from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BlurBaselineConfig(_messages.Message):
    """Config for blur baseline. When enabled, a linear path from the maximally
  blurred image to the input image is created. Using a blurred baseline
  instead of zero (black image) is motivated by the BlurIG approach explained
  here: https://arxiv.org/abs/2004.03383

  Fields:
    maxBlurSigma: The standard deviation of the blur kernel for the blurred
      baseline. The same blurring parameter is used for both the height and
      the width dimension. If not set, the method defaults to the zero (i.e.
      black for images) baseline.
  """
    maxBlurSigma = _messages.FloatField(1, variant=_messages.Variant.FLOAT)