from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.scc import securitycenter_client
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util as scc_util
def ValidateMutexOnSourceAndParent(args):
    """Validates that only a full resource name or split arguments are provided."""
    if '/' in args.source and args.parent is not None:
        raise errors.InvalidSCCInputError('Only provide a full resource name (organizations/123/sources/456) or a --parent flag, not both.')