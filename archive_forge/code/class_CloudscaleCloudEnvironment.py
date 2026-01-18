from __future__ import annotations
import configparser
from ....util import (
from ....config import (
from . import (
class CloudscaleCloudEnvironment(CloudEnvironment):
    """Cloudscale cloud environment plugin. Updates integration test environment after delegation."""

    def get_environment_config(self) -> CloudEnvironmentConfig:
        """Return environment configuration for use in the test environment after delegation."""
        parser = configparser.ConfigParser()
        parser.read(self.config_path)
        env_vars = dict(CLOUDSCALE_API_TOKEN=parser.get('default', 'cloudscale_api_token'))
        display.sensitive.add(env_vars['CLOUDSCALE_API_TOKEN'])
        ansible_vars = dict(cloudscale_resource_prefix=self.resource_prefix)
        ansible_vars.update(dict(((key.lower(), value) for key, value in env_vars.items())))
        return CloudEnvironmentConfig(env_vars=env_vars, ansible_vars=ansible_vars)