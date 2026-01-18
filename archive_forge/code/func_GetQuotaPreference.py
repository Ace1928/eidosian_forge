from apitools.base.py import list_pager
from googlecloudsdk.api_lib.quotas import message_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import common_args
def GetQuotaPreference(args):
    """Retrieve the QuotaPreference for a project, folder or organization.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked with.

  Returns:
    The request QuotaPreference.
  """
    consumer = message_util.CreateConsumer(args.project, args.folder, args.organization)
    client = _GetClientInstance()
    messages = client.MESSAGES_MODULE
    name = _CONSUMER_LOCATION_RESOURCE % consumer + '/quotaPreferences/%s' % args.PREFERENCE_ID
    if args.project:
        request = messages.CloudquotasProjectsLocationsQuotaPreferencesGetRequest(name=name)
        return client.projects_locations_quotaPreferences.Get(request)
    if args.folder:
        request = messages.CloudquotasFoldersLocationsQuotaPreferencesGetRequest(name=name)
        return client.folders_locations_quotaPreferences.Get(request)
    if args.organization:
        request = messages.CloudquotasOrganizationsLocationsQuotaPreferencesGetRequest(name=name)
        return client.organizations_locations_quotaPreferences.Get(request)