from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicedirectoryProjectsLocationsNamespacesWorkloadsPatchRequest(_messages.Message):
    """A ServicedirectoryProjectsLocationsNamespacesWorkloadsPatchRequest
  object.

  Fields:
    name: Immutable. The resource name for the workload in the format
      `projects/*/locations/*/namespaces/*/workloads/*`.
    updateMask: Required. List of fields to be updated in this request.
      Allowable fields: `display_name`, `annotations`. -- Internal
      integrations may update other fields
    workload: A Workload resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    workload = _messages.MessageField('Workload', 3)