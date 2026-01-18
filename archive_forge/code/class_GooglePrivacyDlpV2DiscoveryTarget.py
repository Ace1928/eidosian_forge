from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DiscoveryTarget(_messages.Message):
    """Target used to match against for Discovery.

  Fields:
    bigQueryTarget: BigQuery target for Discovery. The first target to match a
      table will be the one applied.
    cloudSqlTarget: Cloud SQL target for Discovery. The first target to match
      a table will be the one applied.
  """
    bigQueryTarget = _messages.MessageField('GooglePrivacyDlpV2BigQueryDiscoveryTarget', 1)
    cloudSqlTarget = _messages.MessageField('GooglePrivacyDlpV2CloudSqlDiscoveryTarget', 2)