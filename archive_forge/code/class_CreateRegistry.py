from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zunclient.common import utils as zun_utils
from zunclient import exceptions as exc
from zunclient.i18n import _
class CreateRegistry(command.ShowOne):
    """Create a registry"""
    log = logging.getLogger(__name__ + '.CreateRegistry')

    def get_parser(self, prog_name):
        parser = super(CreateRegistry, self).get_parser(prog_name)
        parser.add_argument('--name', metavar='<name>', help='The name of the registry.')
        parser.add_argument('--username', metavar='<username>', help='The username to login to the registry.')
        parser.add_argument('--password', metavar='<password>', help='The password to login to the registry.')
        parser.add_argument('--domain', metavar='<domain>', required=True, help='The domain of the registry.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        opts = {}
        opts['name'] = parsed_args.name
        opts['domain'] = parsed_args.domain
        opts['username'] = parsed_args.username
        opts['password'] = parsed_args.password
        opts = zun_utils.remove_null_parms(**opts)
        registry = client.registries.create(**opts)
        return zip(*sorted(registry._info['registry'].items()))