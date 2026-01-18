from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.config_management import utils
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.policycontroller import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
def _parse_config_sync(configmanagement, msg):
    """Load ConfigSync configuration with the parsed configmanagement yaml.

  Args:
    configmanagement: dict, The data loaded from the config-management.yaml
      given by user.
    msg: The Hub messages package.

  Returns:
    config_sync: The ConfigSync configuration holds configmanagement.spec.git
    or configmanagement.spec.oci being used in MembershipConfigs
  Raises: Error, if required fields are missing from .spec or unsupported fields
    are included in .spec
  """
    if 'spec' not in configmanagement or utils.CONFIG_SYNC not in configmanagement['spec']:
        return None
    spec_source = configmanagement['spec'][utils.CONFIG_SYNC]
    for field in spec_source:
        if field not in yaml.load(utils.APPLY_SPEC_VERSION_1)['spec'][utils.CONFIG_SYNC]:
            raise exceptions.Error('The field .spec.{}.{}'.format(utils.CONFIG_SYNC, field) + ' is unrecognized in this applySpecVersion. Please remove.')
    config_sync = msg.ConfigManagementConfigSync()
    config_sync.enabled = True
    if 'enabled' in spec_source:
        config_sync.enabled = spec_source['enabled']
    source_type = spec_source.get('sourceType', 'git')
    if source_type == 'oci':
        config_sync.oci = _parse_oci_config(spec_source, msg)
    else:
        config_sync.git = _parse_git_config(spec_source, msg)
    if 'sourceFormat' in spec_source:
        config_sync.sourceFormat = spec_source['sourceFormat']
    if 'preventDrift' in spec_source:
        config_sync.preventDrift = spec_source['preventDrift']
    if 'metricsGcpServiceAccountEmail' in spec_source:
        config_sync.metricsGcpServiceAccountEmail = spec_source['metricsGcpServiceAccountEmail']
    return config_sync