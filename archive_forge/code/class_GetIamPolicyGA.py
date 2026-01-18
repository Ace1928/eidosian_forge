from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dns import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dns import flags
from googlecloudsdk.command_lib.iam import iam_util
@base.ReleaseTracks(base.ReleaseTrack.GA)
class GetIamPolicyGA(base.Command):
    """Get the IAM policy for a Cloud DNS managed-zone.

  This command displays the IAM policy of the specified managed-zone.

  ## EXAMPLES

  To view the details of your managed-zone IAM policy , run:

    $ {command} my-zone
  """

    @staticmethod
    def Args(parser):
        flags.GetZoneResourceArg('The name of the managed-zone to get the IAM policy for.').AddToParser(parser)

    def Run(self, args):
        api_version = 'v2'
        dns_client = util.GetApiClient(api_version)
        messages = apis.GetMessagesModule('dns', api_version)
        zone_ref = args.CONCEPTS.zone.Parse()
        resource_name = 'projects/{0}/locations/{1}/managedZones/{2}'.format(zone_ref.project, 'global', zone_ref.managedZone)
        req = messages.DnsManagedZonesGetIamPolicyRequest(resource=resource_name, googleIamV1GetIamPolicyRequest=messages.GoogleIamV1GetIamPolicyRequest(options=messages.GoogleIamV1GetPolicyOptions(requestedPolicyVersion=iam_util.MAX_LIBRARY_IAM_SUPPORTED_VERSION)))
        return dns_client.managedZones.GetIamPolicy(req)