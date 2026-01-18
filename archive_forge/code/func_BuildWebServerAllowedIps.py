from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer.flags import CONNECTION_TYPE_FLAG_ALPHA
from googlecloudsdk.command_lib.composer.flags import CONNECTION_TYPE_FLAG_BETA
from googlecloudsdk.command_lib.composer.flags import CONNECTION_TYPE_FLAG_GA
from googlecloudsdk.command_lib.composer.flags import ENVIRONMENT_SIZE_ALPHA
from googlecloudsdk.command_lib.composer.flags import ENVIRONMENT_SIZE_BETA
from googlecloudsdk.command_lib.composer.flags import ENVIRONMENT_SIZE_GA
def BuildWebServerAllowedIps(allowed_ip_list, allow_all, deny_all):
    """Returns the list of IP ranges that will be sent to the API.

  The resulting IP range list is determined by the options specified in
  environment create or update flags.

  Args:
    allowed_ip_list: [{string: string}], list of IP ranges with descriptions.
    allow_all: bool, True if allow all flag was set.
    deny_all: bool, True if deny all flag was set.

  Returns:
    [{string: string}]: list of IP ranges that will be sent to the API, taking
        into account the values of allow all and deny all flags.
  """
    if deny_all:
        return []
    if allow_all:
        return [{'ip_range': '0.0.0.0/0', 'description': 'Allows access from all IPv4 addresses (default value)'}, {'ip_range': '::0/0', 'description': 'Allows access from all IPv6 addresses (default value)'}]
    return allowed_ip_list