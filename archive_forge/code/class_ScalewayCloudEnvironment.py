from __future__ import annotations
import configparser
from ....util import (
from ....config import (
from . import (
class ScalewayCloudEnvironment(CloudEnvironment):
    """Updates integration test environment after delegation. Will setup the config file as parameter."""

    def get_environment_config(self) -> CloudEnvironmentConfig:
        """Return environment configuration for use in the test environment after delegation."""
        parser = configparser.ConfigParser()
        parser.read(self.config_path)
        env_vars = dict(SCW_API_KEY=parser.get('default', 'key'), SCW_ORG=parser.get('default', 'org'))
        display.sensitive.add(env_vars['SCW_API_KEY'])
        ansible_vars = dict(scw_org=parser.get('default', 'org'))
        return CloudEnvironmentConfig(env_vars=env_vars, ansible_vars=ansible_vars)