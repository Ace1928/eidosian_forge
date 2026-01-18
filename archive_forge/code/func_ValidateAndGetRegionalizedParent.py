from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util as scc_util
def ValidateAndGetRegionalizedParent(args, parent):
    """Appends location to parent."""
    if args.location is not None:
        if '/' in args.location:
            pattern = re.compile('^locations/.*$')
            if not pattern.match(args.location):
                raise errors.InvalidSCCInputError("When providing a full resource path, it must include the pattern '^locations/.*$'.")
            else:
                return f'{parent}/{args.location}'
        else:
            return f'{parent}/locations/{args.location}'