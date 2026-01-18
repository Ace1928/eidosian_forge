from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesProjectsGlobalNetworksGetVpcServiceControlsRequest(_messages.Message):
    """A
  ServicenetworkingServicesProjectsGlobalNetworksGetVpcServiceControlsRequest
  object.

  Fields:
    name: Required. Name of the VPC Service Controls config to retrieve in the
      format:
      `services/{service}/projects/{project}/global/networks/{network}`.
      {service} is the peering service that is managing connectivity for the
      service producer's organization. For Google services that support this
      functionality, this value is `servicenetworking.googleapis.com`.
      {project} is a project number e.g. `12345` that contains the service
      consumer's VPC network. {network} is the name of the service consumer's
      VPC network.
  """
    name = _messages.StringField(1, required=True)