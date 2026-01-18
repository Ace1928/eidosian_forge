from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.scc import securitycenter_client
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util as scc_util
def ValidateLocationAndGetRegionalizedParent(args, parent):
    """Appends location to parent."""
    if args.location:
        if '/' in args.location:
            pattern = re.compile('^locations/[A-Za-z0-9-]{0,61}$')
            if not pattern.match(args.location):
                raise errors.InvalidSCCInputError("When providing a full resource path, it must include the pattern '^locations/.*$'.")
            else:
                return f'{parent}/{args.location}'
        else:
            return f'{parent}/locations/{args.location}'