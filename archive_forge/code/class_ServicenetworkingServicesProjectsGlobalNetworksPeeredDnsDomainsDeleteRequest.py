from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesProjectsGlobalNetworksPeeredDnsDomainsDeleteRequest(_messages.Message):
    """A
  ServicenetworkingServicesProjectsGlobalNetworksPeeredDnsDomainsDeleteRequest
  object.

  Fields:
    name: Required. The name of the peered DNS domain to delete in the format:
      `services/{service}/projects/{project}/global/networks/{network}/peeredD
      nsDomains/{name}`. {service} is the peering service that is managing
      connectivity for the service producer's organization. For Google
      services that support this functionality, this value is
      `servicenetworking.googleapis.com`. {project} is the number of the
      project that contains the service consumer's VPC network e.g. `12345`.
      {network} is the name of the service consumer's VPC network. {name} is
      the name of the peered DNS domain.
  """
    name = _messages.StringField(1, required=True)