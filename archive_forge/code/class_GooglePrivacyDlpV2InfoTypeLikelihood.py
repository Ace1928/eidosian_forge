from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2InfoTypeLikelihood(_messages.Message):
    """Configuration for setting a minimum likelihood per infotype. Used to
  customize the minimum likelihood level for specific infotypes in the
  request. For example, use this if you want to lower the precision for
  PERSON_NAME without lowering the precision for the other infotypes in the
  request.

  Enums:
    MinLikelihoodValueValuesEnum: Only returns findings equal to or above this
      threshold. This field is required or else the configuration fails.

  Fields:
    infoType: Type of information the likelihood threshold applies to. Only
      one likelihood per info_type should be provided. If InfoTypeLikelihood
      does not have an info_type, the configuration fails.
    minLikelihood: Only returns findings equal to or above this threshold.
      This field is required or else the configuration fails.
  """

    class MinLikelihoodValueValuesEnum(_messages.Enum):
        """Only returns findings equal to or above this threshold. This field is
    required or else the configuration fails.

    Values:
      LIKELIHOOD_UNSPECIFIED: Default value; same as POSSIBLE.
      VERY_UNLIKELY: Highest chance of a false positive.
      UNLIKELY: High chance of a false positive.
      POSSIBLE: Some matching signals. The default value.
      LIKELY: Low chance of a false positive.
      VERY_LIKELY: Confidence level is high. Lowest chance of a false
        positive.
    """
        LIKELIHOOD_UNSPECIFIED = 0
        VERY_UNLIKELY = 1
        UNLIKELY = 2
        POSSIBLE = 3
        LIKELY = 4
        VERY_LIKELY = 5
    infoType = _messages.MessageField('GooglePrivacyDlpV2InfoType', 1)
    minLikelihood = _messages.EnumField('MinLikelihoodValueValuesEnum', 2)