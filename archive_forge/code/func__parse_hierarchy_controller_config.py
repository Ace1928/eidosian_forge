from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.config_management import utils
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.policycontroller import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
def _parse_hierarchy_controller_config(configmanagement, msg):
    """Load HierarchyController with the parsed config-management.yaml.

  Args:
    configmanagement: dict, The data loaded from the config-management.yaml
      given by user.
    msg: The Hub messages package.

  Returns:
    hierarchy_controller: The Hierarchy Controller configuration for
    MembershipConfigs, filled in the data parsed from
    configmanagement.spec.hierarchyController
  Raises: Error, if Hierarchy Controller `enabled` set to false but also has
    other fields present in the config
  """
    if 'spec' not in configmanagement or 'hierarchyController' not in configmanagement['spec']:
        return None
    spec = configmanagement['spec']['hierarchyController']
    if spec is None or 'enabled' not in spec:
        raise exceptions.Error('Missing required field .spec.hierarchyController.enabled')
    enabled = spec['enabled']
    if not isinstance(enabled, bool):
        raise exceptions.Error('hierarchyController.enabled should be `true` or `false`')
    config_proto = msg.ConfigManagementHierarchyControllerConfig()
    for field in spec:
        if field not in ['enabled', 'enablePodTreeLabels', 'enableHierarchicalResourceQuota']:
            raise exceptions.Error('Please remove illegal field .spec.hierarchyController{}'.format(field))
        setattr(config_proto, field, spec[field])
    return config_proto