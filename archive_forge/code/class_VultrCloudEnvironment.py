from __future__ import annotations
import configparser
from ....util import (
from ....config import (
from . import (
class VultrCloudEnvironment(CloudEnvironment):
    """Updates integration test environment after delegation. Will setup the config file as parameter."""

    def get_environment_config(self) -> CloudEnvironmentConfig:
        """Return environment configuration for use in the test environment after delegation."""
        parser = configparser.ConfigParser()
        parser.read(self.config_path)
        env_vars = dict(VULTR_API_KEY=parser.get('default', 'key'))
        display.sensitive.add(env_vars['VULTR_API_KEY'])
        ansible_vars = dict(vultr_resource_prefix=self.resource_prefix)
        return CloudEnvironmentConfig(env_vars=env_vars, ansible_vars=ansible_vars)