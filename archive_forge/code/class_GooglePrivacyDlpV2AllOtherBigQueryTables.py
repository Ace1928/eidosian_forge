from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2AllOtherBigQueryTables(_messages.Message):
    """Catch-all for all other tables not specified by other filters. Should
  always be last, except for single-table configurations, which will only have
  a TableReference target.
  """