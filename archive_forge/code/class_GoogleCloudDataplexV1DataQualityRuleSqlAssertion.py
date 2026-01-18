from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataQualityRuleSqlAssertion(_messages.Message):
    """Queries for rows returned by the provided SQL statement. If any rows are
  are returned, this rule fails.The SQL statement needs to use BigQuery
  standard SQL syntax, and must not contain any semicolons.${data()} can be
  used to reference the rows being evaluated, i.e. the table after all
  additional filters (row filters, incremental data filters, sampling) are
  applied.Example: SELECT * FROM ${data()} WHERE price < 0

  Fields:
    sqlStatement: Optional. The SQL statement.
  """
    sqlStatement = _messages.StringField(1)