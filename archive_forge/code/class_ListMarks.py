from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.scc import securitycenter_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.scc import flags as scc_flags
from googlecloudsdk.command_lib.scc import util as scc_util
from googlecloudsdk.command_lib.scc.findings import flags
from googlecloudsdk.command_lib.scc.findings import util
from googlecloudsdk.core.util import times
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class ListMarks(base.ListCommand):
    """List a finding's security marks."""
    detailed_help = {'DESCRIPTION': "List a finding's security marks.", 'EXAMPLES': '\n        List all security marks for `testFinding` under organization `123456` and\n        source `5678`:\n\n          $ {command} `testFinding` --organization=123456 --source=5678\n\n        List all security marks for `testFinding` under project `example-project`\n        and source `5678`:\n\n          $ {command} projects/example-project/sources/5678/findings/testFinding\n\n        List all security marks for `testFinding` under folder `456` and source\n        `5678`:\n\n          $ {command} folders/456/sources/5678/findings/testFinding\n\n        List all security marks for `testFinding` under organization `123456`,\n        source `5678` and `location=eu`:\n\n          $ {command} `testFinding` --organization=123456 --source=5678\n            --location=eu', 'API REFERENCE': '\n      This command uses the Security Command Center API. For more information,\n      see [Security Command Center API.](https://cloud.google.com/security-command-center/docs/reference/rest)'}

    @staticmethod
    def Args(parser):
        base.URI_FLAG.RemoveFromParser(parser)
        flags.CreateFindingArg().AddToParser(parser)
        scc_flags.PAGE_TOKEN_FLAG.AddToParser(parser)
        scc_flags.READ_TIME_FLAG.AddToParser(parser)
        scc_flags.API_VERSION_FLAG.AddToParser(parser)
        scc_flags.LOCATION_FLAG.AddToParser(parser)

    def Run(self, args):
        version = _GetApiVersion(args)
        messages = securitycenter_client.GetMessages(version)
        request = messages.SecuritycenterOrganizationsSourcesFindingsListRequest()
        request.pageToken = args.page_token
        if version == 'v1' and args.IsKnownAndSpecified('read_time'):
            request.readTime = args.read_time
            read_time_dt = times.ParseDateTime(request.readTime)
            request.readTime = times.FormatDateTime(read_time_dt)
        request = _GenerateParent(args, request, version)
        client = securitycenter_client.GetClient(version)
        list_findings_response = client.organizations_sources_findings.List(request)
        response = util.ExtractSecurityMarksFromResponse(list_findings_response.listFindingsResults, args)
        return response