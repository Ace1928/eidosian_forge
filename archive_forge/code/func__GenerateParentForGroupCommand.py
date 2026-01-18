from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
import sys
from googlecloudsdk.api_lib.scc import securitycenter_client
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.scc import flags as scc_flags
from googlecloudsdk.command_lib.scc import util as scc_util
from googlecloudsdk.command_lib.scc.findings import flags
from googlecloudsdk.command_lib.scc.findings import util
from googlecloudsdk.core.util import times
def _GenerateParentForGroupCommand(args, req, version='v1'):
    """Generate a finding's name and parent using org, source and finding id."""
    util.ValidateMutexOnSourceAndParent(args)
    req.groupFindingsRequest.filter = args.filter
    args.filter = ''
    region_resource_patern = re.compile('(organizations|projects|folders)/[a-z0-9]+/sources/[0-9-]{0,62}/locations/[A-Za-z0-9-]{0,62}$')
    parent = scc_util.GetParentFromPositionalArguments(args)
    if region_resource_patern.match(parent):
        req.parent = parent
        return req
    resource_pattern = re.compile('(organizations|projects|folders)/[a-z0-9]+/sources/[0-9-]{0,62}$')
    if resource_pattern.match(parent):
        args.source = parent
    req.parent = util.GetFullSourceName(args, version)
    return req