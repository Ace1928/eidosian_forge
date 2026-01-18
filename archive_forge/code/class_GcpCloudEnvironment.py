from __future__ import annotations
import configparser
from ....util import (
from ....config import (
from . import (
class GcpCloudEnvironment(CloudEnvironment):
    """GCP cloud environment plugin. Updates integration test environment after delegation."""

    def get_environment_config(self) -> CloudEnvironmentConfig:
        """Return environment configuration for use in the test environment after delegation."""
        parser = configparser.ConfigParser()
        parser.read(self.config_path)
        ansible_vars = dict(resource_prefix=self.resource_prefix)
        ansible_vars.update(dict(parser.items('default')))
        return CloudEnvironmentConfig(ansible_vars=ansible_vars)