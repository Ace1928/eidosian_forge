from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Scheduling(_messages.Message):
    """All parameters related to queuing and scheduling of custom jobs.

  Fields:
    disableRetries: Optional. Indicates if the job should retry for internal
      errors after the job starts running. If true, overrides
      `Scheduling.restart_job_on_worker_restart` to false.
    restartJobOnWorkerRestart: Restarts the entire CustomJob if a worker gets
      restarted. This feature can be used by distributed training jobs that
      are not resilient to workers leaving and joining a job.
    timeout: The maximum job running time. The default is 7 days.
  """
    disableRetries = _messages.BooleanField(1)
    restartJobOnWorkerRestart = _messages.BooleanField(2)
    timeout = _messages.StringField(3)