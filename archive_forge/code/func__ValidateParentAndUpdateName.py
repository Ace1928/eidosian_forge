from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.scc import securitycenter_client
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.scc import flags as scc_flags
from googlecloudsdk.command_lib.scc import util as scc_util
from googlecloudsdk.command_lib.scc.findings import flags
from googlecloudsdk.command_lib.scc.findings import util
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import times
def _ValidateParentAndUpdateName(args, req, version):
    """Generate a security mark's name using org, source and finding id."""
    util.ValidateMutexOnFindingAndSourceAndOrganization(args)
    req.name = util.GetFullFindingName(args, version) + '/securityMarks'
    return req