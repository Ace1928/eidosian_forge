from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2NumericalStatsConfig(_messages.Message):
    """Compute numerical stats over an individual column, including min, max,
  and quantiles.

  Fields:
    field: Field to compute numerical stats on. Supported types are integer,
      float, date, datetime, timestamp, time.
  """
    field = _messages.MessageField('GooglePrivacyDlpV2FieldId', 1)