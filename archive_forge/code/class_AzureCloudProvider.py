from __future__ import annotations
import configparser
import typing as t
from ....util import (
from ....config import (
from ....target import (
from ....core_ci import (
from . import (
class AzureCloudProvider(CloudProvider):
    """Azure cloud provider plugin. Sets up cloud resources before delegation."""

    def __init__(self, args: IntegrationConfig) -> None:
        super().__init__(args)
        self.aci: t.Optional[AnsibleCoreCI] = None
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
        if not self._use_static_config():
            self._setup_dynamic()
        get_config(self.config_path)

    def cleanup(self) -> None:
        """Clean up the cloud resource and any temporary configuration files after tests complete."""
        if self.aci:
            self.aci.stop()
        super().cleanup()

    def _setup_dynamic(self) -> None:
        """Request Azure credentials through ansible-core-ci."""
        display.info('Provisioning %s cloud environment.' % self.platform, verbosity=1)
        config = self._read_config_template()
        response = {}
        aci = self._create_ansible_core_ci()
        aci_result = aci.start()
        if not self.args.explain:
            response = aci_result['azure']
            self.aci = aci
        if not self.args.explain:
            values = dict(AZURE_CLIENT_ID=response['clientId'], AZURE_SECRET=response['clientSecret'], AZURE_SUBSCRIPTION_ID=response['subscriptionId'], AZURE_TENANT=response['tenantId'], RESOURCE_GROUP=response['resourceGroupNames'][0], RESOURCE_GROUP_SECONDARY=response['resourceGroupNames'][1])
            display.sensitive.add(values['AZURE_SECRET'])
            config = '\n'.join(('%s: %s' % (key, values[key]) for key in sorted(values)))
            config = '[default]\n' + config
        self._write_config(config)

    def _create_ansible_core_ci(self) -> AnsibleCoreCI:
        """Return an Azure instance of AnsibleCoreCI."""
        return AnsibleCoreCI(self.args, CloudResource(platform='azure'))