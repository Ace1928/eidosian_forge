from cliff import lister
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class UnsetPool(command.Command):
    """Clear pool settings"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('pool', metavar='<pool>', help='Pool to modify (name or ID).')
        parser.add_argument('--name', action='store_true', help='Clear the pool name.')
        parser.add_argument('--description', action='store_true', help='Clear the description of this pool.')
        parser.add_argument('--ca-tls-container-ref', action='store_true', help='Clear the certificate authority certificate reference on this pool.')
        parser.add_argument('--crl-container-ref', action='store_true', help='Clear the certificate revocation list reference on this pool.')
        parser.add_argument('--session-persistence', action='store_true', help='Disables session persistence on the pool.')
        parser.add_argument('--tls-container-ref', action='store_true', help='Clear the certificate reference for this pool.')
        parser.add_argument('--tls-versions', action='store_true', help='Clear all TLS versions from the pool.')
        parser.add_argument('--tls-ciphers', action='store_true', help='Clear all TLS ciphers from the pool.')
        parser.add_argument('--wait', action='store_true', help='Wait for action to complete.')
        parser.add_argument('--alpn-protocols', action='store_true', help='Clear all ALPN protocols from the pool.')
        _tag.add_tag_option_to_parser_for_unset(parser, 'pool')
        return parser

    def take_action(self, parsed_args):
        unset_args = v2_utils.get_unsets(parsed_args)
        if not unset_args and (not parsed_args.all_tag):
            return
        pool_id = v2_utils.get_resource_id(self.app.client_manager.load_balancer.pool_list, 'pools', parsed_args.pool)
        v2_utils.set_tags_for_unset(self.app.client_manager.load_balancer.pool_show, pool_id, unset_args, clear_tags=parsed_args.all_tag)
        body = {'pool': unset_args}
        self.app.client_manager.load_balancer.pool_set(pool_id, json=body)
        if parsed_args.wait:
            v2_utils.wait_for_active(status_f=self.app.client_manager.load_balancer.pool_show, res_id=pool_id)