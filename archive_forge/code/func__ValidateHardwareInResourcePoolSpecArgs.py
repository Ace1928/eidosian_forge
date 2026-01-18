from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai import util as api_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import validation
def _ValidateHardwareInResourcePoolSpecArgs(resource_pool_specs, version):
    """Validates the hardware related fields specified in `--resource-pool-spec` flags.

  Args:
    resource_pool_specs: List[dict], a list of resource pool specs specified via
      command line arguments.
    version: str, the API version this command will interact with, either GA or
      BETA.
  """
    for spec in resource_pool_specs:
        if spec:
            if 'machine-type' not in spec:
                raise exceptions.InvalidArgumentException('--resource-pool-spec', 'Key [machine-type] required in dict arg but not provided.')
            if 'min-replica-count' in spec and 'max-replica-count' not in spec:
                raise exceptions.InvalidArgumentException('--resource-pool-spec', 'Key [max-replica-count] required in dict arg when key [min-replica-count] is provided.')
            if 'max-replica-count' in spec and 'min-replica-count' not in spec:
                raise exceptions.InvalidArgumentException('--resource-pool-spec', 'Key [min-replica-count] required in dict arg when key [max-replica-count] is provided.')
            if 'replica-count' not in spec and 'min-replica-count' not in spec:
                raise exceptions.InvalidArgumentException('--resource-pool-spec', 'Key [replica-count] required in dict arg but not provided.')
            if ('accelerator-count' in spec) != ('accelerator-type' in spec):
                raise exceptions.InvalidArgumentException('--resource-pool-spec', 'Key [accelerator-type] and [accelerator-count] are required to ' + 'use accelerators.')
            accelerator_type = spec.get('accelerator-type', None)
            if accelerator_type:
                type_enum = api_util.GetMessage('MachineSpec', version).AcceleratorTypeValueValuesEnum
                valid_types = [type for type in type_enum.names() if type.startswith('NVIDIA')]
                if accelerator_type not in valid_types:
                    raise exceptions.InvalidArgumentException('--resource-pool-spec', 'Found invalid value of [accelerator-type]: {actual}. Available values are [{expected}].'.format(actual=accelerator_type, expected=', '.join((v for v in sorted(valid_types)))))