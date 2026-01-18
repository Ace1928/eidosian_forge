from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.scc import securitycenter_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.scc import flags as scc_flags
from googlecloudsdk.command_lib.scc import util as scc_util
from googlecloudsdk.command_lib.scc.findings import util
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.ALPHA)
class BulkMute(base.Command):
    """Bulk mute Security Command Center findings based on a filter."""
    detailed_help = {'DESCRIPTION': 'Bulk mute Security Command Center findings based on a filter.', 'EXAMPLES': '\n      To bulk mute findings given organization ``123\'\' based on a filter on\n      category that equals ``XSS_SCRIPTING\'\', run:\n\n        $ {command} --organization=organizations/123\n          --filter="category=\\"XSS_SCRIPTING\\""\n\n      To bulk mute findings given folder ``123\'\' based on a filter on category\n      that equals ``XSS_SCRIPTING\'\', run:\n\n        $ {command} --folder=folders/123 --filter="category=\\"XSS_SCRIPTING\\""\n\n      To bulk mute findings given project ``123\'\' based on a filter on category\n      that equals ``XSS_SCRIPTING\'\', run:\n\n        $ {command} --project=projects/123\n          --filter="category=\\"XSS_SCRIPTING\\""\n\n      To bulk mute findings given organization ``123\'\' based on a filter on\n      category that equals ``XSS_SCRIPTING\'\' and `location=eu` run:\n\n        $ {command} --organization=organizations/123\n          --filter="category=\\"XSS_SCRIPTING\\"" --location=locations/eu\n      ', 'API REFERENCE': '\n      This command uses the Security Command Center API. For more information,\n      see [Security Command Center API.](https://cloud.google.com/security-command-center/docs/reference/rest)'}

    @staticmethod
    def Args(parser):
        parent_group = parser.add_group(mutex=True, required=True)
        parent_group.add_argument('--organization', help="\n        Organization where the findings reside. Formatted as\n        ``organizations/123'' or just ``123''.")
        parent_group.add_argument('--folder', help="\n        Folder where the findings reside. Formatted as ``folders/456'' or just\n        ``456''.")
        parent_group.add_argument('--project', help="\n        Project (id or number) where the findings reside. Formatted as\n        ``projects/789'' or just ``789''.")
        parser.add_argument('--filter', help='The filter string which will applied to findings being muted.')
        scc_flags.API_VERSION_FLAG.AddToParser(parser)
        scc_flags.LOCATION_FLAG.AddToParser(parser)

    def Run(self, args):
        version = scc_util.GetVersionFromArguments(args)
        messages = securitycenter_client.GetMessages(version)
        request = messages.SecuritycenterOrganizationsFindingsBulkMuteRequest()
        request.bulkMuteFindingsRequest = messages.BulkMuteFindingsRequest(filter=args.filter)
        request.parent = util.ValidateAndGetParent(args)
        args.filter = ''
        if version == 'v2':
            request.parent = util.ValidateLocationAndGetRegionalizedParent(args, request.parent)
        client = securitycenter_client.GetClient(version)
        return client.organizations_findings.BulkMute(request)