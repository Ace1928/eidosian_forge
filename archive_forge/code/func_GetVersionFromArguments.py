from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.core import properties
def GetVersionFromArguments(args, resource_name='', deprecated_args=None, version_specific_existing_resource: bool=False):
    """Returns the correct version to call based on the user supplied arguments.

  Args:
    args: arguments
    resource_name: (optional) resource name e.g. finding, mute_config
    deprecated_args: (optional) list of deprecated arguments for a command
    version_specific_existing_resource: (optional) command is invoked on a
      resource which is not interoperable between versions.

  Returns:
    Version of securitycenter api to handle command, either "v1" or "v2"
  """
    location_specified = IsLocationSpecified(args, resource_name)
    if version_specific_existing_resource:
        if location_specified:
            return 'v2'
        else:
            return 'v1'
    if deprecated_args:
        for argument in deprecated_args:
            if args.IsKnownAndSpecified(argument) and location_specified:
                raise errors.InvalidSCCInputError('Location is not available when deprecated arguments are used')
            if args.IsKnownAndSpecified(argument) and (not location_specified):
                return 'v1'
    if args.api_version == 'v1':
        if location_specified:
            return 'v2'
        return 'v1'
    return 'v2'