from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1QueryAssetsOutputConfigBigQueryDestination(_messages.Message):
    """BigQuery destination.

  Fields:
    dataset: Required. The BigQuery dataset where the query results will be
      saved. It has the format of "projects/{projectId}/datasets/{datasetId}".
    table: Required. The BigQuery table where the query results will be saved.
      If this table does not exist, a new table with the given name will be
      created.
    writeDisposition: Specifies the action that occurs if the destination
      table or partition already exists. The following values are supported: *
      WRITE_TRUNCATE: If the table or partition already exists, BigQuery
      overwrites the entire table or all the partitions data. * WRITE_APPEND:
      If the table or partition already exists, BigQuery appends the data to
      the table or the latest partition. * WRITE_EMPTY: If the table already
      exists and contains data, a 'duplicate' error is returned in the job
      result. The default value is WRITE_EMPTY.
  """
    dataset = _messages.StringField(1)
    table = _messages.StringField(2)
    writeDisposition = _messages.StringField(3)