from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.memberships import errors
def GetConnectGatewayServiceName(endpoint_override, location):
    """Get the appropriate Connect Gateway endpoint.

  This function checks the environment endpoint overide configuration for
  Fleet and uses it to determine which Connect Gateway endpoint to use.
  The overridden Fleet value will look like
  https://autopush-gkehub.sandbox.googleapis.com/.

  When there is no override set, this command will return a Connect Gateway
  prod endpoint. When an override is set, an appropriate non-prod endpoint
  will be provided instead.

  For example, when the overridden value looks like
  https://autopush-gkehub.sandbox.googleapis.com/,
  the function will return
  autopush-connectgateway.sandbox.googleapis.com.

  Regional prefixes are supported via the location argument. For example, when
  the overridden value looks like
  https://autopush-gkehub.sandbox.googleapis.com/ and location is passed in as
  "us-west1", the function will return
  us-west1-autopush-connectgateway.sandbox.googleapis.com.

  Args:
    endpoint_override: The URL set as the API endpoint override for 'gkehub'.
      None if the override is not set.
    location: The location against which the command is supposed to run. This
      will be used to dynamically modify the service name to a location-specific
      value. If this is the value 'global' or None, a global service name is
      returned.

  Returns:
    The service name to use for this command invocation, optionally modified
    to target a specific location.

  Raises:
    UnknownApiEndpointOverrideError if the Fleet API endpoint override is not
    one of the standard values.
  """
    prefix = '' if location in ('global', None) else '{}-'.format(location)
    if not endpoint_override or endpoint_override == 'https://gkehub.googleapis.com/':
        return '{}connectgateway.googleapis.com'.format(prefix)
    elif 'autopush-gkehub' in endpoint_override:
        return '{}autopush-connectgateway.sandbox.googleapis.com'.format(prefix)
    elif 'staging-gkehub' in endpoint_override:
        return '{}staging-connectgateway.sandbox.googleapis.com'.format(prefix)
    else:
        raise errors.UnknownApiEndpointOverrideError('gkehub')