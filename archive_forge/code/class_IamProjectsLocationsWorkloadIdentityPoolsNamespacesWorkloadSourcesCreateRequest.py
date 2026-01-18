from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsNamespacesWorkloadSourcesCreateRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsNamespacesWorkloadSourcesCrea
  teRequest object.

  Fields:
    parent: Required. The parent resource to create the workload source in.
    workloadSource: A WorkloadSource resource to be passed as the request
      body.
    workloadSourceId: Required. The ID to use for the workload source, which
      becomes the final component of the resource name. If ID of the
      WorkloadSource resource determines which workloads may be matched. The
      following formats are supported: - `project-{project_number}` matches
      workloads within the referenced Google Cloud project.
  """
    parent = _messages.StringField(1, required=True)
    workloadSource = _messages.MessageField('WorkloadSource', 2)
    workloadSourceId = _messages.StringField(3)