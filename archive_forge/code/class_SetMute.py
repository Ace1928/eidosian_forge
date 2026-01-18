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
class SetMute(base.Command):
    """Update a Security Command Center finding's mute state."""
    detailed_help = {'DESCRIPTION': "Update a Security Command Center finding's mute state.", 'EXAMPLES': "\n        To update finding's mute state to ``MUTED'', given finding\n        `organizations/123/sources/456/findings/789`, run:\n\n          $ {command} 789 --organization=organizations/123 --source=456\n            --mute=MUTED\n\n        To update finding's mute state to ``UNMUTED'', given finding\n        `organizations/123/sources/456/findings/789`, run:\n\n          $ {command} 789 --organization=organizations/123 --source=456\n            --mute=UNMUTED\n\n        To update finding's mute state to ``MUTED'', given finding\n        `folders/123/sources/456/findings/789`, run:\n\n          $ {command} 789 --folder=folders/123 --source=456 --mute=MUTED\n\n        To update finding's mute state to ``MUTED'', given finding\n        `projects/123/sources/456/findings/789`, run:\n\n          $ {command} 789 --project=projects/123 --source=456 --mute=MUTED\n\n        To update finding's mute state to ``MUTED'', given finding\n        `organizations/123/sources/456/findings/789` and `location=eu`, run:\n\n          $ {command} 789 --organization=organizations/123 --source=456\n          --mute=MUTED --location=locations/eu", 'API REFERENCE': '\n      This command uses the Security Command Center API. For more information,\n      see [Security Command Center API.](https://cloud.google.com/security-command-center/docs/reference/rest)'}

    @staticmethod
    def Args(parser):
        scc_flags.API_VERSION_FLAG.AddToParser(parser)
        scc_flags.LOCATION_FLAG.AddToParser(parser)
        parser.add_argument('finding', help='ID of the finding or the full resource name of the finding.')
        source_group = parser.add_group(mutex=True)
        source_group.add_argument('--organization', help="Organization where the finding resides. Formatted as\n        ``organizations/123'' or just ``123''.")
        source_group.add_argument('--folder', help="Folder where the finding resides. Formatted as ``folders/456''\n        or just ``456''.")
        source_group.add_argument('--project', help="Project (id or number) where the finding resides. Formatted as\n        ``projects/789'' or just ``789''.")
        base.ChoiceArgument('--mute', required=True, default='mute_unspecified', choices=['muted', 'unmuted'], help_str='Desired mute state of the finding.').AddToParser(parser)
        parser.add_argument('--source', help='ID of the source.')

    def Run(self, args):
        version = scc_util.GetVersionFromArguments(args, args.finding)
        messages = securitycenter_client.GetMessages(version)
        request = messages.SecuritycenterOrganizationsSourcesFindingsSetMuteRequest()
        request.setMuteRequest = messages.SetMuteRequest()
        set_mute_dict = {'mute_unspecified': messages.SetMuteRequest.MuteValueValuesEnum.MUTE_UNSPECIFIED, 'muted': messages.SetMuteRequest.MuteValueValuesEnum.MUTED, 'unmuted': messages.SetMuteRequest.MuteValueValuesEnum.UNMUTED, 'undefined': messages.SetMuteRequest.MuteValueValuesEnum.UNDEFINED}
        args.mute = args.mute.lower()
        request.setMuteRequest.mute = set_mute_dict.get(args.mute, messages.SetMuteRequest.MuteValueValuesEnum.UNDEFINED)
        parent = util.ValidateAndGetParent(args)
        if parent is not None:
            util.ValidateSourceAndFindingIdIfParentProvided(args)
            if version == 'v1':
                request.name = parent + '/sources/' + args.source + '/findings/' + args.finding
            elif version == 'v2':
                source_parent = parent + '/sources/' + args.source
                regionalized_parent = util.ValidateLocationAndGetRegionalizedParent(args, source_parent)
                request.name = regionalized_parent + '/findings/' + args.finding
        else:
            request.name = util.GetFullFindingName(args, version)
        client = securitycenter_client.GetClient(version)
        response = client.organizations_sources_findings.SetMute(request)
        return response