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
def UpdateAssetSecurityMarksReqHook(ref, args, req):
    """Generate a security mark's name using org, source and finding."""
    del ref
    _ValidateMutexOnAssetAndOrganization(args)
    req.name = _GetAssetNameForParent(args) + '/securityMarks'
    if req.updateMask is not None:
        req.updateMask = CleanUpUserInput(req.updateMask)
    return req