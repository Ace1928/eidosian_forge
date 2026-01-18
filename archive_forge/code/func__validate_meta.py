from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.config_management import utils
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.policycontroller import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
def _validate_meta(configmanagement):
    """Validate the parsed configmanagement yaml.

  Args:
    configmanagement: The dict loaded from yaml.
  """
    if not isinstance(configmanagement, dict):
        raise exceptions.Error('Invalid ConfigManagement template.')
    if configmanagement.get('applySpecVersion') != 1:
        raise exceptions.Error('Only "applySpecVersion: 1" is supported. To use a later version,please fetch the config by running\ngcloud container fleet config-management fetch-for-apply')
    if 'spec' not in configmanagement:
        raise exceptions.Error('Missing required field .spec')
    spec = configmanagement['spec']
    legal_fields = {utils.CONFIG_SYNC, utils.POLICY_CONTROLLER, utils.HNC, utils.CLUSTER, utils.UPGRADES}
    for field in spec:
        if field not in legal_fields:
            raise exceptions.Error('Please remove illegal field .spec.{}'.format(field))