from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsBatchesDeleteRequest(_messages.Message):
    """A DataprocProjectsLocationsBatchesDeleteRequest object.

  Fields:
    name: Required. The fully qualified name of the batch to retrieve in the
      format "projects/PROJECT_ID/locations/DATAPROC_REGION/batches/BATCH_ID"
  """
    name = _messages.StringField(1, required=True)