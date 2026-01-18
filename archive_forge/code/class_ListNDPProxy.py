import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class ListNDPProxy(command.Lister):
    _description = _('List NDP proxies')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--router', metavar='<router>', help=_('List only NDP proxies belonging to this router (name or ID)'))
        parser.add_argument('--port', metavar='<port>', help=_('List only NDP proxies associated to this port (name or ID)'))
        parser.add_argument('--ip-address', metavar='ip-address', help=_('List only NDP proxies according to their IPv6 address'))
        parser.add_argument('--project', metavar='<project>', help=_('List NDP proxies according to their project (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('List NDP proxies according to their name'))
        identity_common.add_project_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        identity_client = self.app.client_manager.identity
        columns = ('id', 'name', 'router_id', 'ip_address', 'project_id')
        headers = ('ID', 'Name', 'Router ID', 'IP Address', 'Project')
        query = {}
        if parsed_args.router:
            router = client.find_router(parsed_args.router, ignore_missing=False)
            query['router_id'] = router.id
        if parsed_args.port:
            port = client.find_port(parsed_args.port, ignore_missing=False)
            query['port_id'] = port.id
        if parsed_args.ip_address is not None:
            query['ip_address'] = parsed_args.ip_address
        if parsed_args.project:
            project_id = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
            query['project_id'] = project_id
        if parsed_args.name:
            query['name'] = parsed_args.name
        data = client.ndp_proxies(**query)
        return (headers, (utils.get_item_properties(s, columns, formatters={}) for s in data))