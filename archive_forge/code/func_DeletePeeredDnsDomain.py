from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.util import apis
def DeletePeeredDnsDomain(project_number, service, network, name):
    """Make API call to delete a peered DNS domain for a specific connection.

  Args:
    project_number: The number of the project which is peered with the service.
    service: The name of the service to delete a peered DNS domain for.
    network: The network in the consumer project peered with the service.
    name: The name of the peered DNS domain.

  Raises:
    exceptions.DeletePeeredDnsDomainPermissionDeniedException: when the delete
    peered DNS domain API fails.
    apitools_exceptions.HttpError: Another miscellaneous error with the peering
    service.

  Returns:
    The result of the delete peered DNS domain operation.
  """
    client = _GetClientInstance()
    messages = client.MESSAGES_MODULE
    request = messages.ServicenetworkingServicesProjectsGlobalNetworksPeeredDnsDomainsDeleteRequest(name='services/%s/projects/%s/global/networks/%s/peeredDnsDomains/%s' % (service, project_number, network, name))
    try:
        return client.services_projects_global_networks_peeredDnsDomains.Delete(request)
    except (apitools_exceptions.HttpForbiddenError, apitools_exceptions.HttpNotFoundError) as e:
        exceptions.ReraiseError(e, exceptions.DeletePeeredDnsDomainPermissionDeniedException)