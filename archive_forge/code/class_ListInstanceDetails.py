from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.os_config import resource_args
from googlecloudsdk.core.resource import resource_projector
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class ListInstanceDetails(base.ListCommand):
    """List the instance details for an OS patch job.

  ## EXAMPLES

  To list the instance details for each instance in the patch job `job1`, run:

        $ {command} job1

  """

    @staticmethod
    def Args(parser):
        base.URI_FLAG.RemoveFromParser(parser)
        resource_args.AddPatchJobResourceArg(parser, 'to list instance details.')
        parser.display_info.AddFormat('\n          table(\n            name.basename(),\n            zone,\n            state,\n            failure_reason()\n          )\n        ')
        parser.display_info.AddTransforms({'failure_reason': _TransformFailureReason})

    def Run(self, args):
        patch_job_ref = args.CONCEPTS.patch_job.Parse()
        release_track = self.ReleaseTrack()
        client = osconfig_api_utils.GetClientInstance(release_track)
        messages = osconfig_api_utils.GetClientMessages(release_track)
        request = messages.OsconfigProjectsPatchJobsInstanceDetailsListRequest(pageSize=args.page_size, parent=patch_job_ref.RelativeName())
        results = list(list_pager.YieldFromList(client.projects_patchJobs_instanceDetails, request, limit=args.limit, batch_size=args.page_size, field='patchJobInstanceDetails', batch_size_attribute='pageSize'))
        return _PostProcessListResult(results)