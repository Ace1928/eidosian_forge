from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai import util as api_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import validation
def _ValidateResourcePoolSpecArgs(resource_pool_specs, version):
    """Validates the argument values specified via `--resource-pool-spec` flags.

  Args:
    resource_pool_specs: List[dict], a list of resource pool specs specified via
      command line arguments.
    version: str, the API version this command will interact with, either GA or
      BETA.
  """
    if not resource_pool_specs[0]:
        raise exceptions.InvalidArgumentException('--resource-pool-spec', 'Empty value is not allowed for the first `--resource-pool-spec` flag.')
    _ValidateHardwareInResourcePoolSpecArgs(resource_pool_specs, version)