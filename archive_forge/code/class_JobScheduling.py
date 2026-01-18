from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobScheduling(_messages.Message):
    """Job scheduling options.

  Fields:
    maxFailuresPerHour: Optional. Maximum number of times per hour a driver
      can be restarted as a result of driver exiting with non-zero code before
      job is reported failed.A job might be reported as thrashing if the
      driver exits with a non-zero code four times within a 10-minute
      window.Maximum value is 10.Note: This restartable job option is not
      supported in Dataproc workflow templates
      (https://cloud.google.com/dataproc/docs/concepts/workflows/using-
      workflows#adding_jobs_to_a_template).
    maxFailuresTotal: Optional. Maximum total number of times a driver can be
      restarted as a result of the driver exiting with a non-zero code. After
      the maximum number is reached, the job will be reported as
      failed.Maximum value is 240.Note: Currently, this restartable job option
      is not supported in Dataproc workflow templates
      (https://cloud.google.com/dataproc/docs/concepts/workflows/using-
      workflows#adding_jobs_to_a_template).
  """
    maxFailuresPerHour = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    maxFailuresTotal = _messages.IntegerField(2, variant=_messages.Variant.INT32)