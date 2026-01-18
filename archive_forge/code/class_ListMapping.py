import json
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListMapping(command.Lister):
    _description = _('List mappings')

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        data = identity_client.federation.mappings.list()
        columns = ('ID', 'schema_version')
        items = [utils.get_item_properties(s, columns) for s in data]
        return (columns, items)