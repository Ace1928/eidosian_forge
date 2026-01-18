from apitools.base.py import list_pager
from googlecloudsdk.api_lib.quotas import message_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import common_args
def ListQuotaPreferences(args):
    """Lists QuotaPreferences in a given project, folder or organization.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked with.

  Returns:
    List of QuotaPreferences.
  """
    consumer = message_util.CreateConsumer(args.project, args.folder, args.organization)
    client = _GetClientInstance()
    messages = client.MESSAGES_MODULE
    parent = _CONSUMER_LOCATION_RESOURCE % consumer
    print(args.page_size)
    if args.project:
        request = messages.CloudquotasProjectsLocationsQuotaPreferencesListRequest(parent=parent, pageSize=args.page_size, pageToken=args.page_token, filter=_GetFilter(args.filter, args.reconciling_only), orderBy=common_args.ParseSortByArg(args.sort_by))
        return list_pager.YieldFromList(client.projects_locations_quotaPreferences, request, batch_size_attribute='pageSize', batch_size=args.page_size if args.page_size is not None else PAGE_SIZE, field='quotaPreferences')
    if args.folder:
        request = messages.CloudquotasFoldersLocationsQuotaPreferencesListRequest(parent=parent, pageSize=args.page_size, pageToken=args.page_token, filter=_GetFilter(args.filter, args.reconciling_only), orderBy=common_args.ParseSortByArg(args.sort_by))
        return list_pager.YieldFromList(client.folders_locations_quotaPreferences, request, batch_size_attribute='pageSize', batch_size=args.page_size if args.page_size is not None else PAGE_SIZE, field='quotaPreferences')
    if args.organization:
        request = messages.CloudquotasOrganizationsLocationsQuotaPreferencesListRequest(parent=parent, pageSize=args.page_size, pageToken=args.page_token, filter=_GetFilter(args.filter, args.reconciling_only), orderBy=common_args.ParseSortByArg(args.sort_by))
        return list_pager.YieldFromList(client.organizations_locations_quotaPreferences, request, batch_size_attribute='pageSize', batch_size=args.page_size if args.page_size is not None else PAGE_SIZE, field='quotaPreferences')