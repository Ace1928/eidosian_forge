from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuleTypeValueValuesEnum(_messages.Enum):
    """The type of the data quality rule.

    Values:
      RULE_TYPE_UNSPECIFIED: An unspecified rule type.
      NON_NULL_EXPECTATION: Please see https://cloud.google.com/dataplex/docs/
        reference/rest/v1/DataQualityRule#nonnullexpectation.
      RANGE_EXPECTATION: Please see https://cloud.google.com/dataplex/docs/ref
        erence/rest/v1/DataQualityRule#rangeexpectation.
      REGEX_EXPECTATION: Please see https://cloud.google.com/dataplex/docs/ref
        erence/rest/v1/DataQualityRule#regexexpectation.
      ROW_CONDITION_EXPECTATION: Please see https://cloud.google.com/dataplex/
        docs/reference/rest/v1/DataQualityRule#rowconditionexpectation.
      SET_EXPECTATION: Please see https://cloud.google.com/dataplex/docs/refer
        ence/rest/v1/DataQualityRule#setexpectation.
      STATISTIC_RANGE_EXPECTATION: Please see https://cloud.google.com/dataple
        x/docs/reference/rest/v1/DataQualityRule#statisticrangeexpectation.
      TABLE_CONDITION_EXPECTATION: Please see https://cloud.google.com/dataple
        x/docs/reference/rest/v1/DataQualityRule#tableconditionexpectation.
      UNIQUENESS_EXPECTATION: Please see https://cloud.google.com/dataplex/doc
        s/reference/rest/v1/DataQualityRule#uniquenessexpectation.
    """
    RULE_TYPE_UNSPECIFIED = 0
    NON_NULL_EXPECTATION = 1
    RANGE_EXPECTATION = 2
    REGEX_EXPECTATION = 3
    ROW_CONDITION_EXPECTATION = 4
    SET_EXPECTATION = 5
    STATISTIC_RANGE_EXPECTATION = 6
    TABLE_CONDITION_EXPECTATION = 7
    UNIQUENESS_EXPECTATION = 8