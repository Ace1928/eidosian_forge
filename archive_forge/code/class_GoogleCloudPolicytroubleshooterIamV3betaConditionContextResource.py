from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3betaConditionContextResource(_messages.Message):
    """Core attributes for a resource. A resource is an addressable (named)
  entity provided by the destination service. For example, a Compute Engine
  instance.

  Fields:
    name: The stable identifier (name) of a resource on the `service`. A
      resource can be logically identified as
      `//{resource.service}/{resource.name}`. Unlike the resource URI, the
      resource name doesn't contain any protocol and version information. For
      a list of full resource name formats, see
      https://cloud.google.com/iam/help/troubleshooter/full-resource-names
    service: The name of the service that this resource belongs to, such as
      `compute.googleapis.com`. The service name might not match the DNS
      hostname that actually serves the request. For a full list of resource
      service values, see
      https://cloud.google.com/iam/help/conditions/resource-services
    type: The type of the resource, in the format `{service}/{kind}`. For a
      full list of resource type values, see
      https://cloud.google.com/iam/help/conditions/resource-types
  """
    name = _messages.StringField(1)
    service = _messages.StringField(2)
    type = _messages.StringField(3)