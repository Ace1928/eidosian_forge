from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zunclient.common import utils as zun_utils
from zunclient import exceptions as exc
from zunclient.i18n import _
class DeleteRegistry(command.Command):
    """Delete specified registry(s)"""
    log = logging.getLogger(__name__ + '.Deleteregistry')

    def get_parser(self, prog_name):
        parser = super(DeleteRegistry, self).get_parser(prog_name)
        parser.add_argument('registry', metavar='<registry>', nargs='+', help='ID or name of the registry(s) to delete.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        registries = parsed_args.registry
        for registry in registries:
            opts = {}
            opts['id'] = registry
            opts = zun_utils.remove_null_parms(**opts)
            try:
                client.registries.delete(**opts)
                print(_('Request to delete registry %s has been accepted.') % registry)
            except Exception as e:
                print('Delete for registry %(registry)s failed: %(e)s' % {'registry': registry, 'e': e})