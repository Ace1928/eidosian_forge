from tempest.lib.cli import base
from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests import client
from designateclient.functionaltests import config
class BaseDesignateTest(base.ClientTestBase):

    def _get_clients(self):
        config.read_config()
        return client.DesignateCLI.as_user('default')

    def ensure_tld_exists(self, tld):
        try:
            self.clients.as_user('admin').tld_create(tld)
        except CommandFailed:
            pass

    def _is_entity_in_list(self, entity, entity_list):
        """Determines if the given entity exists in the given list.

        Uses the id for comparison.

        Certain entities (e.g. zone import, export) cannot be made
        comparable in a list of CLI output results, because the fields
        in a list command can be different from those in a show command.

        """
        return any([entity_record.id == entity.id for entity_record in entity_list])