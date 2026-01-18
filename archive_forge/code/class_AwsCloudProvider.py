from __future__ import annotations
import os
import uuid
import configparser
import typing as t
from ....util import (
from ....config import (
from ....target import (
from ....core_ci import (
from ....host_configs import (
from . import (
class AwsCloudProvider(CloudProvider):
    """AWS cloud provider plugin. Sets up cloud resources before delegation."""

    def __init__(self, args: IntegrationConfig) -> None:
        super().__init__(args)
        self.uses_config = True

    def filter(self, targets: tuple[IntegrationTarget, ...], exclude: list[str]) -> None:
        """Filter out the cloud tests when the necessary config and resources are not available."""
        aci = self._create_ansible_core_ci()
        if aci.available:
            return
        super().filter(targets, exclude)

    def setup(self) -> None:
        """Setup the cloud resource before delegation and register a cleanup callback."""
        super().setup()
        aws_config_path = os.path.expanduser('~/.aws')
        if os.path.exists(aws_config_path) and isinstance(self.args.controller, OriginConfig):
            raise ApplicationError('Rename "%s" or use the --docker or --remote option to isolate tests.' % aws_config_path)
        if not self._use_static_config():
            self._setup_dynamic()

    def _setup_dynamic(self) -> None:
        """Request AWS credentials through the Ansible Core CI service."""
        display.info('Provisioning %s cloud environment.' % self.platform, verbosity=1)
        config = self._read_config_template()
        aci = self._create_ansible_core_ci()
        response = aci.start()
        if not self.args.explain:
            credentials = response['aws']['credentials']
            values = dict(ACCESS_KEY=credentials['access_key'], SECRET_KEY=credentials['secret_key'], SECURITY_TOKEN=credentials['session_token'], REGION='us-east-1')
            display.sensitive.add(values['SECRET_KEY'])
            display.sensitive.add(values['SECURITY_TOKEN'])
            config = self._populate_config_template(config, values)
        self._write_config(config)

    def _create_ansible_core_ci(self) -> AnsibleCoreCI:
        """Return an AWS instance of AnsibleCoreCI."""
        return AnsibleCoreCI(self.args, CloudResource(platform='aws'))