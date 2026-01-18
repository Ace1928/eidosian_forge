import os
import os_client_config
from tempest.lib.cli import base
class MistralCLIAltAuth(base.ClientTestBase):
    _mistral_url = None

    def _get_alt_clients(self):
        creds = credentials('devstack-alt-member')
        clients = base.CLIClient(username=creds['username'], password=creds['password'], project_name=creds['project_name'], tenant_name=creds['project_name'], user_domain_id=creds['user_domain_id'], project_domain_id=creds['project_domain_id'], uri=creds['auth_url'], cli_dir=CLI_DIR)
        return clients

    def _get_clients(self):
        return self._get_alt_clients()

    def mistral_alt(self, action, flags='', params='', mode='alt_user'):
        """Executes Mistral command for alt_user from alt_tenant."""
        mistral_url_op = '--os-mistral-url %s' % self._mistral_url
        flags = '{} --insecure'.format(flags)
        return self.clients.cmd_with_auth('mistral %s' % mistral_url_op, action, flags, params)