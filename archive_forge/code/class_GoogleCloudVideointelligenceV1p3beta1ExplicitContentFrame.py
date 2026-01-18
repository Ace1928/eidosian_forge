from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p3beta1ExplicitContentFrame(_messages.Message):
    """Video frame level annotation results for explicit content.

  Enums:
    PornographyLikelihoodValueValuesEnum: Likelihood of the pornography
      content..

  Fields:
    pornographyLikelihood: Likelihood of the pornography content..
    timeOffset: Time-offset, relative to the beginning of the video,
      corresponding to the video frame for this location.
  """

    class PornographyLikelihoodValueValuesEnum(_messages.Enum):
        """Likelihood of the pornography content..

    Values:
      LIKELIHOOD_UNSPECIFIED: Unspecified likelihood.
      VERY_UNLIKELY: Very unlikely.
      UNLIKELY: Unlikely.
      POSSIBLE: Possible.
      LIKELY: Likely.
      VERY_LIKELY: Very likely.
    """
        LIKELIHOOD_UNSPECIFIED = 0
        VERY_UNLIKELY = 1
        UNLIKELY = 2
        POSSIBLE = 3
        LIKELY = 4
        VERY_LIKELY = 5
    pornographyLikelihood = _messages.EnumField('PornographyLikelihoodValueValuesEnum', 1)
    timeOffset = _messages.StringField(2)