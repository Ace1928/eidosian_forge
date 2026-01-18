from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zunclient.common import utils as zun_utils
from zunclient import exceptions as exc
from zunclient.i18n import _
class UpdateRegistry(command.ShowOne):
    """Update one or more attributes of the registry"""
    log = logging.getLogger(__name__ + '.UpdateRegistry')

    def get_parser(self, prog_name):
        parser = super(UpdateRegistry, self).get_parser(prog_name)
        parser.add_argument('registry', metavar='<registry>', help='ID or name of the registry to update.')
        parser.add_argument('--username', metavar='<username>', help='The new username of registry to update.')
        parser.add_argument('--password', metavar='<password>', help='The new password of registry to update.')
        parser.add_argument('--name', metavar='<name>', help='The new name of registry to update.')
        parser.add_argument('--domain', metavar='<domain>', help='The new domain of registry to update.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        registry = parsed_args.registry
        opts = {}
        opts['username'] = parsed_args.username
        opts['password'] = parsed_args.password
        opts['domain'] = parsed_args.domain
        opts['name'] = parsed_args.name
        opts = zun_utils.remove_null_parms(**opts)
        if not opts:
            raise exc.CommandError('You must update at least one property')
        registry = client.registries.update(registry, **opts)
        return zip(*sorted(registry._info['registry'].items()))