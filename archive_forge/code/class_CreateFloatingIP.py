import logging
from blazarclient import command
class CreateFloatingIP(command.CreateCommand):
    """Create a floating IP."""
    resource = 'floatingip'
    json_indent = 4
    log = logging.getLogger(__name__ + '.CreateFloatingIP')

    def get_parser(self, prog_name):
        parser = super(CreateFloatingIP, self).get_parser(prog_name)
        parser.add_argument('network_id', metavar='NETWORK_ID', help='External network ID to which the floating IP belongs')
        parser.add_argument('floating_ip_address', metavar='FLOATING_IP_ADDRESS', help='Floating IP address to add to Blazar')
        return parser

    def args2body(self, parsed_args):
        params = {}
        if parsed_args.network_id:
            params['network_id'] = parsed_args.network_id
        if parsed_args.floating_ip_address:
            params['floating_ip_address'] = parsed_args.floating_ip_address
        return params