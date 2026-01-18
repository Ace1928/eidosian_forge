from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2InspectConfig(_messages.Message):
    """Configuration description of the scanning process. When used with
  redactContent only info_types and min_likelihood are currently used.

  Enums:
    ContentOptionsValueListEntryValuesEnum:
    MinLikelihoodValueValuesEnum: Only returns findings equal to or above this
      threshold. The default is POSSIBLE. In general, the highest likelihood
      setting yields the fewest findings in results and the lowest chance of a
      false positive. For more information, see [Match
      likelihood](https://cloud.google.com/sensitive-data-
      protection/docs/likelihood).

  Fields:
    contentOptions: Deprecated and unused.
    customInfoTypes: CustomInfoTypes provided by the user. See
      https://cloud.google.com/sensitive-data-protection/docs/creating-custom-
      infotypes to learn more.
    excludeInfoTypes: When true, excludes type information of the findings.
      This is not used for data profiling.
    includeQuote: When true, a contextual quote from the data that triggered a
      finding is included in the response; see Finding.quote. This is not used
      for data profiling.
    infoTypes: Restricts what info_types to look for. The values must
      correspond to InfoType values returned by ListInfoTypes or listed at
      https://cloud.google.com/sensitive-data-protection/docs/infotypes-
      reference. When no InfoTypes or CustomInfoTypes are specified in a
      request, the system may automatically choose a default list of detectors
      to run, which may change over time. If you need precise control and
      predictability as to what detectors are run you should specify specific
      InfoTypes listed in the reference, otherwise a default list will be
      used, which may change over time.
    limits: Configuration to control the number of findings returned. This is
      not used for data profiling. When redacting sensitive data from images,
      finding limits don't apply. They can cause unexpected or inconsistent
      results, where only some data is redacted. Don't include finding limits
      in RedactImage requests. Otherwise, Cloud DLP returns an error. When set
      within an InspectJobConfig, the specified maximum values aren't hard
      limits. If an inspection job reaches these limits, the job ends
      gradually, not abruptly. Therefore, the actual number of findings that
      Cloud DLP returns can be multiple times higher than these maximum
      values.
    minLikelihood: Only returns findings equal to or above this threshold. The
      default is POSSIBLE. In general, the highest likelihood setting yields
      the fewest findings in results and the lowest chance of a false
      positive. For more information, see [Match
      likelihood](https://cloud.google.com/sensitive-data-
      protection/docs/likelihood).
    minLikelihoodPerInfoType: Minimum likelihood per infotype. For each
      infotype, a user can specify a minimum likelihood. The system only
      returns a finding if its likelihood is above this threshold. If this
      field is not set, the system uses the InspectConfig min_likelihood.
    ruleSet: Set of rules to apply to the findings for this InspectConfig.
      Exclusion rules, contained in the set are executed in the end, other
      rules are executed in the order they are specified for each info type.
  """

    class ContentOptionsValueListEntryValuesEnum(_messages.Enum):
        """ContentOptionsValueListEntryValuesEnum enum type.

    Values:
      CONTENT_UNSPECIFIED: Includes entire content of a file or a data stream.
      CONTENT_TEXT: Text content within the data, excluding any metadata.
      CONTENT_IMAGE: Images found in the data.
    """
        CONTENT_UNSPECIFIED = 0
        CONTENT_TEXT = 1
        CONTENT_IMAGE = 2

    class MinLikelihoodValueValuesEnum(_messages.Enum):
        """Only returns findings equal to or above this threshold. The default is
    POSSIBLE. In general, the highest likelihood setting yields the fewest
    findings in results and the lowest chance of a false positive. For more
    information, see [Match likelihood](https://cloud.google.com/sensitive-
    data-protection/docs/likelihood).

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
    contentOptions = _messages.EnumField('ContentOptionsValueListEntryValuesEnum', 1, repeated=True)
    customInfoTypes = _messages.MessageField('GooglePrivacyDlpV2CustomInfoType', 2, repeated=True)
    excludeInfoTypes = _messages.BooleanField(3)
    includeQuote = _messages.BooleanField(4)
    infoTypes = _messages.MessageField('GooglePrivacyDlpV2InfoType', 5, repeated=True)
    limits = _messages.MessageField('GooglePrivacyDlpV2FindingLimits', 6)
    minLikelihood = _messages.EnumField('MinLikelihoodValueValuesEnum', 7)
    minLikelihoodPerInfoType = _messages.MessageField('GooglePrivacyDlpV2InfoTypeLikelihood', 8, repeated=True)
    ruleSet = _messages.MessageField('GooglePrivacyDlpV2InspectionRuleSet', 9, repeated=True)