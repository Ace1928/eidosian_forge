from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataQualityRuleResult(_messages.Message):
    """DataQualityRuleResult provides a more detailed, per-rule view of the
  results.

  Fields:
    assertionRowCount: Output only. The number of rows returned by the sql
      statement in the SqlAssertion rule.This field is only valid for
      SqlAssertion rules.
    evaluatedCount: The number of rows a rule was evaluated against.This field
      is only valid for row-level type rules.Evaluated count can be configured
      to either include all rows (default) - with null rows automatically
      failing rule evaluation, or exclude null rows from the evaluated_count,
      by setting ignore_nulls = true.
    failingRowsQuery: The query to find rows that did not pass this rule.This
      field is only valid for row-level type rules.
    nullCount: The number of rows with null values in the specified column.
    passRatio: The ratio of passed_count / evaluated_count.This field is only
      valid for row-level type rules.
    passed: Whether the rule passed or failed.
    passedCount: The number of rows which passed a rule evaluation.This field
      is only valid for row-level type rules.
    rule: The rule specified in the DataQualitySpec, as is.
  """
    assertionRowCount = _messages.IntegerField(1)
    evaluatedCount = _messages.IntegerField(2)
    failingRowsQuery = _messages.StringField(3)
    nullCount = _messages.IntegerField(4)
    passRatio = _messages.FloatField(5)
    passed = _messages.BooleanField(6)
    passedCount = _messages.IntegerField(7)
    rule = _messages.MessageField('GoogleCloudDataplexV1DataQualityRule', 8)