from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataQualityRule(_messages.Message):
    """A rule captures data quality intent about a data source.

  Fields:
    column: Optional. The unnested column which this rule is evaluated
      against.
    description: Optional. Description of the rule. The maximum length is
      1,024 characters.
    dimension: Required. The dimension a rule belongs to. Results are also
      aggregated at the dimension level. Supported dimensions are
      "COMPLETENESS", "ACCURACY", "CONSISTENCY", "VALIDITY", "UNIQUENESS",
      "INTEGRITY"
    ignoreNull: Optional. Rows with null values will automatically fail a
      rule, unless ignore_null is true. In that case, such null rows are
      trivially considered passing.This field is only valid for the following
      type of rules: RangeExpectation RegexExpectation SetExpectation
      UniquenessExpectation
    name: Optional. A mutable name for the rule. The name must contain only
      letters (a-z, A-Z), numbers (0-9), or hyphens (-). The maximum length is
      63 characters. Must start with a letter. Must end with a number or a
      letter.
    nonNullExpectation: Row-level rule which evaluates whether each column
      value is null.
    rangeExpectation: Row-level rule which evaluates whether each column value
      lies between a specified range.
    regexExpectation: Row-level rule which evaluates whether each column value
      matches a specified regex.
    rowConditionExpectation: Row-level rule which evaluates whether each row
      in a table passes the specified condition.
    setExpectation: Row-level rule which evaluates whether each column value
      is contained by a specified set.
    sqlAssertion: Aggregate rule which evaluates the number of rows returned
      for the provided statement.
    statisticRangeExpectation: Aggregate rule which evaluates whether the
      column aggregate statistic lies between a specified range.
    tableConditionExpectation: Aggregate rule which evaluates whether the
      provided expression is true for a table.
    threshold: Optional. The minimum ratio of passing_rows / total_rows
      required to pass this rule, with a range of 0.0, 1.0.0 indicates default
      value (i.e. 1.0).This field is only valid for row-level type rules.
    uniquenessExpectation: Row-level rule which evaluates whether each column
      value is unique.
  """
    column = _messages.StringField(1)
    description = _messages.StringField(2)
    dimension = _messages.StringField(3)
    ignoreNull = _messages.BooleanField(4)
    name = _messages.StringField(5)
    nonNullExpectation = _messages.MessageField('GoogleCloudDataplexV1DataQualityRuleNonNullExpectation', 6)
    rangeExpectation = _messages.MessageField('GoogleCloudDataplexV1DataQualityRuleRangeExpectation', 7)
    regexExpectation = _messages.MessageField('GoogleCloudDataplexV1DataQualityRuleRegexExpectation', 8)
    rowConditionExpectation = _messages.MessageField('GoogleCloudDataplexV1DataQualityRuleRowConditionExpectation', 9)
    setExpectation = _messages.MessageField('GoogleCloudDataplexV1DataQualityRuleSetExpectation', 10)
    sqlAssertion = _messages.MessageField('GoogleCloudDataplexV1DataQualityRuleSqlAssertion', 11)
    statisticRangeExpectation = _messages.MessageField('GoogleCloudDataplexV1DataQualityRuleStatisticRangeExpectation', 12)
    tableConditionExpectation = _messages.MessageField('GoogleCloudDataplexV1DataQualityRuleTableConditionExpectation', 13)
    threshold = _messages.FloatField(14)
    uniquenessExpectation = _messages.MessageField('GoogleCloudDataplexV1DataQualityRuleUniquenessExpectation', 15)