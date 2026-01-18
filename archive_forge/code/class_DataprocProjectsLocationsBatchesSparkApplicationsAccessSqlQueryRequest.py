from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsBatchesSparkApplicationsAccessSqlQueryRequest(_messages.Message):
    """A DataprocProjectsLocationsBatchesSparkApplicationsAccessSqlQueryRequest
  object.

  Fields:
    details: Optional. Lists/ hides details of Spark plan nodes. True is set
      to list and false to hide.
    executionId: Required. Execution ID
    name: Required. The fully qualified name of the batch to retrieve in the
      format "projects/PROJECT_ID/locations/DATAPROC_REGION/batches/BATCH_ID/s
      parkApplications/APPLICATION_ID"
    parent: Required. Parent (Batch) resource reference.
    planDescription: Optional. Enables/ disables physical plan description on
      demand
  """
    details = _messages.BooleanField(1)
    executionId = _messages.IntegerField(2)
    name = _messages.StringField(3, required=True)
    parent = _messages.StringField(4)
    planDescription = _messages.BooleanField(5)