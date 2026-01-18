from __future__ import annotations
import configparser
from ....util import (
from ....config import (
from . import (
class CloudscaleCloudProvider(CloudProvider):
    """Cloudscale cloud provider plugin. Sets up cloud resources before delegation."""

    def __init__(self, args: IntegrationConfig) -> None:
        super().__init__(args)
        self.uses_config = True

    def setup(self) -> None:
        """Setup the cloud resource before delegation and register a cleanup callback."""
        super().setup()
        self._use_static_config()