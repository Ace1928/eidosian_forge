import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
class ListFloatingIPPortForwarding(command.Lister):
    _description = _('List floating IP port forwarding')

    def get_parser(self, prog_name):
        parser = super(ListFloatingIPPortForwarding, self).get_parser(prog_name)
        parser.add_argument('floating_ip', metavar='<floating-ip>', help=_('Floating IP that the port forwarding belongs to (IP address or ID)'))
        parser.add_argument('--port', metavar='<port>', help=_('Filter the list result by the ID or name of the internal network port'))
        parser.add_argument('--external-protocol-port', metavar='<port-number>', dest='external_protocol_port', help=_('Filter the list result by the protocol port number of the floating IP'))
        parser.add_argument('--protocol', metavar='protocol', help=_('Filter the list result by the port protocol'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        columns = ('id', 'internal_port_id', 'internal_ip_address', 'internal_port', 'internal_port_range', 'external_port', 'external_port_range', 'protocol', 'description')
        headers = ('ID', 'Internal Port ID', 'Internal IP Address', 'Internal Port', 'Internal Port Range', 'External Port', 'External Port Range', 'Protocol', 'Description')
        query = {}
        if parsed_args.port:
            port = client.find_port(parsed_args.port, ignore_missing=False)
            query['internal_port_id'] = port.id
        external_port = parsed_args.external_protocol_port
        if external_port:
            if ':' in external_port:
                query['external_port_range'] = external_port
            else:
                query['external_port'] = int(parsed_args.external_protocol_port)
        if parsed_args.protocol is not None:
            query['protocol'] = parsed_args.protocol
        obj = client.find_ip(parsed_args.floating_ip, ignore_missing=False)
        data = client.floating_ip_port_forwardings(obj, **query)
        return (headers, (utils.get_item_properties(s, columns, formatters={}) for s in data))