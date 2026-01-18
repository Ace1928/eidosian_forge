import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowConntrackHelper(command.ShowOne):
    _description = _('Display L3 conntrack helper details')

    def get_parser(self, prog_name):
        parser = super(ShowConntrackHelper, self).get_parser(prog_name)
        parser.add_argument('router', metavar='<router>', help=_('Router that the conntrack helper belong to'))
        parser.add_argument('conntrack_helper_id', metavar='<conntrack-helper-id>', help=_('The ID of the conntrack helper'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        router = client.find_router(parsed_args.router, ignore_missing=False)
        obj = client.get_conntrack_helper(parsed_args.conntrack_helper_id, router.id)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns, formatters={})
        return (display_columns, data)