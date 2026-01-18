from __future__ import annotations
import json
import configparser
import os
import urllib.parse
import typing as t
from ....util import (
from ....config import (
from ....docker_util import (
from ....containers import (
from . import (
def _setup_dynamic(self) -> None:
    """Create a CloudStack simulator using docker."""
    config = self._read_config_template()
    self.port = 8888
    ports = [self.port]
    descriptor = run_support_container(self.args, self.platform, self.image, 'cloudstack-sim', ports)
    if not descriptor:
        return
    docker_exec(self.args, descriptor.name, ['find', '/var/lib/mysql', '-type', 'f', '-exec', 'touch', '{}', ';'], capture=True)
    if self.args.explain:
        values = dict(HOST=self.host, PORT=str(self.port))
    else:
        credentials = self._get_credentials(descriptor.name)
        values = dict(HOST=descriptor.name, PORT=str(self.port), KEY=credentials['apikey'], SECRET=credentials['secretkey'])
        display.sensitive.add(values['SECRET'])
    config = self._populate_config_template(config, values)
    self._write_config(config)