from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesProjectsGlobalNetworksPeeredDnsDomainsCreateRequest(_messages.Message):
    """A
  ServicenetworkingServicesProjectsGlobalNetworksPeeredDnsDomainsCreateRequest
  object.

  Fields:
    parent: Required. Parent resource identifying the connection for which the
      peered DNS domain will be created in the format:
      `services/{service}/projects/{project}/global/networks/{network}`
      {service} is the peering service that is managing connectivity for the
      service producer's organization. For Google services that support this
      functionality, this value is `servicenetworking.googleapis.com`.
      {project} is the number of the project that contains the service
      consumer's VPC network e.g. `12345`. {network} is the name of the
      service consumer's VPC network.
    peeredDnsDomain: A PeeredDnsDomain resource to be passed as the request
      body.
  """
    parent = _messages.StringField(1, required=True)
    peeredDnsDomain = _messages.MessageField('PeeredDnsDomain', 2)