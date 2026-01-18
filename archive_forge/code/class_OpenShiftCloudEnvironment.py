from __future__ import annotations
import re
from ....io import (
from ....util import (
from ....config import (
from ....containers import (
from . import (
class OpenShiftCloudEnvironment(CloudEnvironment):
    """OpenShift cloud environment plugin. Updates integration test environment after delegation."""

    def get_environment_config(self) -> CloudEnvironmentConfig:
        """Return environment configuration for use in the test environment after delegation."""
        env_vars = dict(K8S_AUTH_KUBECONFIG=self.config_path)
        return CloudEnvironmentConfig(env_vars=env_vars)