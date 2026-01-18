from __future__ import annotations
import os
from ....config import (
from ....containers import (
from . import (
class ACMEProvider(CloudProvider):
    """ACME plugin. Sets up cloud resources for tests."""

    def __init__(self, args: IntegrationConfig) -> None:
        super().__init__(args)
        if os.environ.get('ANSIBLE_ACME_CONTAINER'):
            self.image = os.environ.get('ANSIBLE_ACME_CONTAINER')
        else:
            self.image = 'quay.io/ansible/acme-test-container:2.1.0'
        self.uses_docker = True

    def setup(self) -> None:
        """Setup the cloud resource before delegation and register a cleanup callback."""
        super().setup()
        if self._use_static_config():
            self._setup_static()
        else:
            self._setup_dynamic()

    def _setup_dynamic(self) -> None:
        """Create a ACME test container using docker."""
        ports = [5000, 14000]
        descriptor = run_support_container(self.args, self.platform, self.image, 'acme-simulator', ports)
        if not descriptor:
            return
        self._set_cloud_config('acme_host', descriptor.name)

    def _setup_static(self) -> None:
        raise NotImplementedError()