from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataQualityRuleSetExpectation(_messages.Message):
    """Evaluates whether each column value is contained by a specified set.

  Fields:
    values: Optional. Expected values for the column value.
  """
    values = _messages.StringField(1, repeated=True)