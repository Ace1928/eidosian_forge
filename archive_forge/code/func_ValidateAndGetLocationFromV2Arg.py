import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util
def ValidateAndGetLocationFromV2Arg(args):
    """Returns the location from the location arg, or throws an error."""
    if args.IsKnownAndSpecified('location'):
        if '/' in args.location:
            long_pattern = re.compile('^locations/.*$')
            if long_pattern.match(args.location):
                return args.location.split('/')[-1]
            else:
                raise errors.InvalidSCCInputError("When providing a full resource path, it must include the pattern '^locations/.*$'.")
        else:
            return args.location
    else:
        return 'global'