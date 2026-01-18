from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2TransformationDetailsStorageConfig(_messages.Message):
    """Config for storing transformation details.

  Fields:
    table: The BigQuery table in which to store the output. This may be an
      existing table or in a new table in an existing dataset. If table_id is
      not set a new one will be generated for you with the following format:
      dlp_googleapis_transformation_details_yyyy_mm_dd_[dlp_job_id]. Pacific
      time zone will be used for generating the date details.
  """
    table = _messages.MessageField('GooglePrivacyDlpV2BigQueryTable', 1)