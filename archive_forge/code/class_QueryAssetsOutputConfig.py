from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryAssetsOutputConfig(_messages.Message):
    """Output configuration query assets.

  Fields:
    bigqueryDestination: BigQuery destination where the query results will be
      saved.
  """
    bigqueryDestination = _messages.MessageField('GoogleCloudAssetV1QueryAssetsOutputConfigBigQueryDestination', 1)