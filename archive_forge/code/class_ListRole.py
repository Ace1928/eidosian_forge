import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListRole(command.Lister):
    _description = _('List roles')

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        columns = ('ID', 'Name')
        data = identity_client.roles.list()
        return (columns, (utils.get_item_properties(s, columns, formatters={}) for s in data))