from __future__ import annotations
import configparser
from ....util import (
from ....config import (
from ....target import (
from ....core_ci import (
from . import (
class HcloudCloudEnvironment(CloudEnvironment):
    """Hetzner Cloud cloud environment plugin. Updates integration test environment after delegation."""

    def get_environment_config(self) -> CloudEnvironmentConfig:
        """Return environment configuration for use in the test environment after delegation."""
        parser = configparser.ConfigParser()
        parser.read(self.config_path)
        env_vars = dict(HCLOUD_TOKEN=parser.get('default', 'hcloud_api_token'))
        display.sensitive.add(env_vars['HCLOUD_TOKEN'])
        ansible_vars = dict(hcloud_prefix=self.resource_prefix)
        ansible_vars.update(dict(((key.lower(), value) for key, value in env_vars.items())))
        return CloudEnvironmentConfig(env_vars=env_vars, ansible_vars=ansible_vars)