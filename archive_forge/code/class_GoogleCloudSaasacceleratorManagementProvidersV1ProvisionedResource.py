from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSaasacceleratorManagementProvidersV1ProvisionedResource(_messages.Message):
    """Describes provisioned dataplane resources.

  Fields:
    resourceType: Type of the resource. This can be either a GCP resource or a
      custom one (e.g. another cloud provider's VM). For GCP compute resources
      use singular form of the names listed in GCP compute API documentation
      (https://cloud.google.com/compute/docs/reference/rest/v1/), prefixed
      with 'compute-', for example: 'compute-instance', 'compute-disk',
      'compute-autoscaler'.
    resourceUrl: URL identifying the resource, e.g.
      "https://www.googleapis.com/compute/v1/projects/...)".
  """
    resourceType = _messages.StringField(1)
    resourceUrl = _messages.StringField(2)