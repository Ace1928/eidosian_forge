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
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import times
def _V1GenerateRequestArgumentsForCreateCommand(args):
    """Generate the request's finding name, finding ID, parent and v1 Finding using specified arguments.

  Args:
    args: (argparse namespace)

  Returns:
    req: Modified request
  """
    messages = securitycenter_client.GetMessages('v1')
    request = messages.SecuritycenterOrganizationsSourcesFindingsCreateRequest()
    request.finding = messages.Finding(category=args.category, resourceName=args.resource_name, state=util.ConvertStateInput(args.state, 'v1'))
    request.finding.externalUri = args.external_uri
    if args.IsKnownAndSpecified('source_properties'):
        request.finding.sourceProperties = util.ConvertSourceProperties(args.source_properties, 'v1')
    event_time_dt = times.ParseDateTime(args.event_time)
    request.finding.eventTime = times.FormatDateTime(event_time_dt)
    util.ValidateMutexOnFindingAndSourceAndOrganization(args)
    finding_name = util.GetFullFindingName(args, 'v1')
    request.parent = util.GetSourceParentFromFindingName(finding_name, 'v1')
    request.findingId = util.GetFindingIdFromName(finding_name)
    if not request.finding:
        request.finding = messages.Finding()
    request.finding.name = finding_name
    return request