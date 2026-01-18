from __future__ import annotations
import os
from ....config import (
from ....containers import (
from . import (
class NiosEnvironment(CloudEnvironment):
    """NIOS environment plugin. Updates integration test environment after delegation."""

    def get_environment_config(self) -> CloudEnvironmentConfig:
        """Return environment configuration for use in the test environment after delegation."""
        ansible_vars = dict(nios_provider=dict(host=self._get_cloud_config('NIOS_HOST'), username='admin', password='infoblox'))
        return CloudEnvironmentConfig(ansible_vars=ansible_vars)