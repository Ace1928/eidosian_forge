from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.core import properties
def ValidateAndGetLocation(args, version):
    """Validates --location flag input and returns location."""
    if version == 'v2':
        if args.location is not None:
            name_pattern = re.compile('^locations/[A-Za-z0-9-]{0,61}$')
            id_pattern = re.compile('^[A-Za-z0-9-]{0,61}$')
            if name_pattern.match(args.location):
                return args.location.split('/')[1]
            if id_pattern.match(args.location):
                return args.location
            raise errors.InvalidSCCInputError("location does not match the pattern '^locations/[A-Za-z0-9-]{0,61}$'. or [A-Za-z0-9-]{0,61}")
    return args.location