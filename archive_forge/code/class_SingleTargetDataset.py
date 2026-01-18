from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SingleTargetDataset(_messages.Message):
    """A single target dataset to which all data will be streamed.

  Fields:
    datasetId: The dataset ID of the target dataset. DatasetIds allowed
      characters: https://cloud.google.com/bigquery/docs/reference/rest/v2/dat
      asets#datasetreference.
  """
    datasetId = _messages.StringField(1)