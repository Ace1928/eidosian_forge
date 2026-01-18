import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class ListDomain(command.Lister):
    _description = _('List domains')

    def get_parser(self, prog_name):
        parser = super(ListDomain, self).get_parser(prog_name)
        parser.add_argument('--name', metavar='<name>', help=_('The domain name'))
        parser.add_argument('--enabled', dest='enabled', action='store_true', help=_('The domains that are enabled will be returned'))
        return parser

    def take_action(self, parsed_args):
        kwargs = {}
        if parsed_args.name:
            kwargs['name'] = parsed_args.name
        if parsed_args.enabled:
            kwargs['enabled'] = True
        columns = ('ID', 'Name', 'Enabled', 'Description')
        data = self.app.client_manager.identity.domains.list(**kwargs)
        return (columns, (utils.get_item_properties(s, columns, formatters={}) for s in data))