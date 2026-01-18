from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadRegistration(_messages.Message):
    """Message describing WorkloadRegistration object

  Fields:
    createTime: Output only. Time when this WorkloadRegistration resource was
      created.
    name: Output only. Name of this WorkloadRegistration resource. Format:
      `projects/{project ID or number}/locations/{location}
      /workloadRegistrations/{client-defined workload_registration_id}`
      {location} is Fleet membership location for GKE clusters and this is
      subject to change.
    status: Output only. The status of the WorkloadRegistration resource.
    updateTime: Output only. Time when this WorkloadRegistration resource was
      most recently updated.
    workloadSelector: Required. Selects the workloads in the registration.
  """
    createTime = _messages.StringField(1)
    name = _messages.StringField(2)
    status = _messages.MessageField('RegistrationStatus', 3)
    updateTime = _messages.StringField(4)
    workloadSelector = _messages.MessageField('WorkloadSelector', 5)