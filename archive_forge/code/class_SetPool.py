from cliff import lister
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class SetPool(command.Command):
    """Update a pool"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('pool', metavar='<pool>', help='Pool to update (name or ID).')
        parser.add_argument('--name', metavar='<name>', help='Set the name of the pool.')
        parser.add_argument('--description', metavar='<description>', help='Set the description of the pool.')
        parser.add_argument('--session-persistence', metavar='<session_persistence>', help='Set the session persistence for the listener (key=value).')
        parser.add_argument('--lb-algorithm', metavar='{' + ','.join(ALGORITHM_CHOICES) + '}', choices=ALGORITHM_CHOICES, type=lambda s: s.upper(), help='Set the load balancing algorithm to use.')
        admin_group = parser.add_mutually_exclusive_group()
        admin_group.add_argument('--enable', action='store_true', default=None, help='Enable pool.')
        admin_group.add_argument('--disable', action='store_true', default=None, help='Disable pool.')
        parser.add_argument('--tls-container-ref', metavar='<container-ref>', help='The URI to the key manager service secrets container containing the certificate and key for TERMINATED_TLS pools to re-encrpt the traffic from TERMINATED_TLS listener to backend servers.')
        parser.add_argument('--ca-tls-container-ref', metavar='<ca_tls_container_ref>', help='The URI to the key manager service secrets container containing the CA certificate for TERMINATED_TLS listeners to check the backend servers certificates in ssl traffic.')
        parser.add_argument('--crl-container-ref', metavar='<crl_container_ref>', help='The URI to the key manager service secrets container containting the CA revocation list file for TERMINATED_TLS listeners to valid the backend servers certificates in ssl traffic.')
        tls_enable = parser.add_mutually_exclusive_group()
        tls_enable.add_argument('--enable-tls', action='store_true', default=None, help='Enable backend associated members re-encryption.')
        tls_enable.add_argument('--disable-tls', action='store_true', default=None, help='disable backend associated members re-encryption.')
        parser.add_argument('--wait', action='store_true', help='Wait for action to complete.')
        parser.add_argument('--tls-ciphers', metavar='<tls_ciphers>', help='Set the TLS ciphers to be used by the pool in OpenSSL cipher string format.')
        parser.add_argument('--tls-version', dest='tls_versions', metavar='<tls_versions>', nargs='?', action='append', help='Set the TLS protocol version to be used by the pool (can be set multiple times).')
        parser.add_argument('--alpn-protocol', dest='alpn_protocols', metavar='<alpn_protocols>', nargs='?', action='append', help='Set the ALPN protocol to be used by the pool (can be set multiple times).')
        _tag.add_tag_option_to_parser_for_set(parser, 'pool')
        return parser

    def take_action(self, parsed_args):
        attrs = v2_utils.get_pool_attrs(self.app.client_manager, parsed_args)
        pool_id = attrs.pop('pool_id')
        v2_utils.set_tags_for_set(self.app.client_manager.load_balancer.pool_show, pool_id, attrs, clear_tags=parsed_args.no_tag)
        body = {'pool': attrs}
        self.app.client_manager.load_balancer.pool_set(pool_id, json=body)
        if parsed_args.wait:
            v2_utils.wait_for_active(status_f=self.app.client_manager.load_balancer.pool_show, res_id=pool_id)