from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateTransferJobRequest(_messages.Message):
    """Request passed to UpdateTransferJob.

  Fields:
    projectId: Required. The ID of the Google Cloud project that owns the job.
    transferJob: Required. The job to update. `transferJob` is expected to
      specify one or more of five fields: description, transfer_spec,
      notification_config, logging_config, and status. An
      `UpdateTransferJobRequest` that specifies other fields are rejected with
      the error INVALID_ARGUMENT. Updating a job status to DELETED requires
      `storagetransfer.jobs.delete` permission.
    updateTransferJobFieldMask: The field mask of the fields in `transferJob`
      that are to be updated in this request. Fields in `transferJob` that can
      be updated are: description, transfer_spec, notification_config,
      logging_config, and status. To update the `transfer_spec` of the job, a
      complete transfer specification must be provided. An incomplete
      specification missing any required fields is rejected with the error
      INVALID_ARGUMENT.
  """
    projectId = _messages.StringField(1)
    transferJob = _messages.MessageField('TransferJob', 2)
    updateTransferJobFieldMask = _messages.StringField(3)