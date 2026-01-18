from __future__ import annotations
import configparser
from ....util import (
from . import (
class OpenNebulaCloudProvider(CloudProvider):
    """Checks if a configuration file has been passed or fixtures are going to be used for testing"""

    def setup(self) -> None:
        """Setup the cloud resource before delegation and register a cleanup callback."""
        super().setup()
        if not self._use_static_config():
            self._setup_dynamic()
        self.uses_config = True

    def _setup_dynamic(self) -> None:
        display.info('No config file provided, will run test from fixtures')
        config = self._read_config_template()
        values = dict(URL='http://localhost/RPC2', USERNAME='oneadmin', PASSWORD='onepass', FIXTURES='true', REPLAY='true')
        config = self._populate_config_template(config, values)
        self._write_config(config)