import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class ListLocalIPAssociation(command.Lister):
    _description = _('List Local IP Associations')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('local_ip', metavar='<local-ip>', help=_('Local IP that port associations belongs to'))
        parser.add_argument('--fixed-port', metavar='<fixed-port>', help=_('Filter the list result by the ID or name of the fixed port'))
        parser.add_argument('--fixed-ip', metavar='<fixed-ip>', help=_('Filter the list result by fixed ip'))
        parser.add_argument('--host', metavar='<host>', help=_('Filter the list result by given host'))
        identity_common.add_project_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        columns = ('local_ip_id', 'local_ip_address', 'fixed_port_id', 'fixed_ip', 'host')
        column_headers = ('Local IP ID', 'Local IP Address', 'Fixed port ID', 'Fixed IP', 'Host')
        attrs = {}
        obj = client.find_local_ip(parsed_args.local_ip, ignore_missing=False)
        if parsed_args.fixed_port:
            port = client.find_port(parsed_args.fixed_port, ignore_missing=False)
            attrs['fixed_port_id'] = port.id
        if parsed_args.fixed_ip:
            attrs['fixed_ip'] = parsed_args.fixed_ip
        if parsed_args.host:
            attrs['host'] = parsed_args.host
        data = client.local_ip_associations(obj, **attrs)
        return (column_headers, (utils.get_item_properties(s, columns, formatters={}) for s in data))