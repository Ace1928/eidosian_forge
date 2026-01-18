from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resourcesettings import utils as api_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_settings import arguments
from googlecloudsdk.command_lib.resource_settings import utils
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class UnsetValue(base.DescribeCommand):
    """Remove the value of a resource setting.

  Remove the value of a resource setting. This reverts the resource to
  inheriting the resource settings from above it in the resource hierarchy,
  if any is set on a resource above it.

  ## EXAMPLES

  To unset the resource settings ``net-preferredDnsServers'' with the
  project ``foo-project'', run:

    $ {command} net-preferredDnsServers --project=foo-project
  """

    @staticmethod
    def Args(parser):
        arguments.AddSettingsNameArgToParser(parser)
        arguments.AddResourceFlagsToParser(parser)

    def Run(self, args):
        """Unset the resource settings.

    Args:
      args: argparse.Namespace, An object that contains the values for the
        arguments specified in the Args method.

    Returns:
       The deleted settings.
    """
        settings_service = api_utils.GetServiceFromArgs(args)
        setting_path = utils.GetSettingsPathFromArgs(args)
        etag = input.etag if hasattr(input, 'etag') else None
        delete_request = api_utils.GetPatchRequestFromArgs(args, setting_path, None, etag)
        setting_value = settings_service.Patch(delete_request)
        return setting_value