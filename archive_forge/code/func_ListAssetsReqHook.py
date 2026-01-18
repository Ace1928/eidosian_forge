from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.scc import securitycenter_client as sc_client
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.scc.hooks import CleanUpUserInput
from googlecloudsdk.command_lib.scc.hooks import GetOrganization
from googlecloudsdk.command_lib.scc.hooks import GetParentFromResourceName
from googlecloudsdk.command_lib.scc.util import GetParentFromPositionalArguments
def ListAssetsReqHook(ref, args, req):
    """Hook up filter such that the CSCC filter is used rather than gcloud."""
    del ref
    req.parent = GetParentFromPositionalArguments(args)
    if req.fieldMask is not None:
        req.fieldMask = CleanUpUserInput(req.fieldMask)
    req.filter = args.filter
    args.filter = ''
    return req