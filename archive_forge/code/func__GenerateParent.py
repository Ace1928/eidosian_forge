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
def _GenerateParent(args, req, version):
    """Generates a finding's parent and adds filter based on finding name."""
    util.ValidateMutexOnFindingAndSourceAndOrganization(args)
    finding_name = util.GetFullFindingName(args, version)
    req.parent = util.GetSourceParentFromFindingName(finding_name, version)
    req.filter = f'name : "{util.GetFindingIdFromName(finding_name)}"'
    return req