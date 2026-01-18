from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.scc import securitycenter_client
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util as scc_util
def GetApiVersionUsingDeprecatedArgs(args, deprecated_args):
    """Determine what version to call from --location and --api-version."""
    if not args.parent:
        parent = scc_util.GetParentFromPositionalArguments(args)
    else:
        parent = args.parent
    return scc_util.GetVersionFromArguments(args, parent, deprecated_args)