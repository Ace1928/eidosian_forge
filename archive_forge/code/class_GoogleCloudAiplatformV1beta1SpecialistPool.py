from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SpecialistPool(_messages.Message):
    """SpecialistPool represents customers' own workforce to work on their data
  labeling jobs. It includes a group of specialist managers and workers.
  Managers are responsible for managing the workers in this pool as well as
  customers' data labeling jobs associated with this pool. Customers create
  specialist pool as well as start data labeling jobs on Cloud, managers and
  workers handle the jobs using CrowdCompute console.

  Fields:
    displayName: Required. The user-defined name of the SpecialistPool. The
      name can be up to 128 characters long and can consist of any UTF-8
      characters. This field should be unique on project-level.
    name: Required. The resource name of the SpecialistPool.
    pendingDataLabelingJobs: Output only. The resource name of the pending
      data labeling jobs.
    specialistManagerEmails: The email addresses of the managers in the
      SpecialistPool.
    specialistManagersCount: Output only. The number of managers in this
      SpecialistPool.
    specialistWorkerEmails: The email addresses of workers in the
      SpecialistPool.
  """
    displayName = _messages.StringField(1)
    name = _messages.StringField(2)
    pendingDataLabelingJobs = _messages.StringField(3, repeated=True)
    specialistManagerEmails = _messages.StringField(4, repeated=True)
    specialistManagersCount = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    specialistWorkerEmails = _messages.StringField(6, repeated=True)