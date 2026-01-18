from apitools.base.py import list_pager
from googlecloudsdk.api_lib.quotas import message_util
from googlecloudsdk.api_lib.util import apis
def GetQuotaInfo(args):
    """Retrieve the QuotaInfo of a quota for a project, folder or organization.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked with.

  Returns:
    The request QuotaInfo
  """
    consumer = message_util.CreateConsumer(args.project, args.folder, args.organization)
    client = _GetClientInstance()
    messages = client.MESSAGES_MODULE
    name = _CONSUMER_LOCATION_SERVICE_RESOURCE % (consumer, args.service) + '/quotaInfos/%s' % args.QUOTA_ID
    if args.project:
        request = messages.CloudquotasProjectsLocationsServicesQuotaInfosGetRequest(name=name)
        return client.projects_locations_services_quotaInfos.Get(request)
    if args.folder:
        request = messages.CloudquotasFoldersLocationsServicesQuotaInfosGetRequest(name=name)
        return client.folders_locations_services_quotaInfos.Get(request)
    if args.organization:
        request = messages.CloudquotasOrganizationsLocationsServicesQuotaInfosGetRequest(name=name)
        return client.organizations_locations_services_quotaInfos.Get(request)