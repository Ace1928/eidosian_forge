from apitools.base.py import list_pager
from googlecloudsdk.api_lib.quotas import message_util
from googlecloudsdk.api_lib.util import apis
def ListQuotaInfo(args):
    """Lists info for all quotas for a given project, folder or organization.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked with.

  Returns:
    List of QuotaInfo
  """
    consumer = message_util.CreateConsumer(args.project, args.folder, args.organization)
    client = _GetClientInstance()
    messages = client.MESSAGES_MODULE
    parent = _CONSUMER_LOCATION_SERVICE_RESOURCE % (consumer, args.service)
    if args.project:
        request = messages.CloudquotasProjectsLocationsServicesQuotaInfosListRequest(parent=parent, pageSize=args.page_size, pageToken=args.page_token)
        return list_pager.YieldFromList(client.projects_locations_services_quotaInfos, request, batch_size_attribute='pageSize', batch_size=args.page_size if args.page_size is not None else PAGE_SIZE, field='quotaInfos')
    if args.folder:
        request = messages.CloudquotasFoldersLocationsServicesQuotaInfosListRequest(parent=parent, pageSize=args.page_size, pageToken=args.page_token)
        return list_pager.YieldFromList(client.folders_locations_services_quotaInfos, request, batch_size_attribute='pageSize', batch_size=args.page_size if args.page_size is not None else PAGE_SIZE, field='quotaInfos')
    if args.organization:
        request = messages.CloudquotasOrganizationsLocationsServicesQuotaInfosListRequest(parent=parent, pageSize=args.page_size, pageToken=args.page_token)
        return list_pager.YieldFromList(client.organizations_locations_services_quotaInfos, request, batch_size_attribute='pageSize', batch_size=args.page_size if args.page_size is not None else PAGE_SIZE, field='quotaInfos')