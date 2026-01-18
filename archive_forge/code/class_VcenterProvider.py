from __future__ import annotations
import configparser
from ....util import (
from ....config import (
from . import (
class VcenterProvider(CloudProvider):
    """VMware vcenter/esx plugin. Sets up cloud resources for tests."""

    def __init__(self, args: IntegrationConfig) -> None:
        super().__init__(args)
        self.uses_config = True

    def setup(self) -> None:
        """Setup the cloud resource before delegation and register a cleanup callback."""
        super().setup()
        if not self._use_static_config():
            raise ApplicationError('Configuration file does not exist: %s' % self.config_static_path)