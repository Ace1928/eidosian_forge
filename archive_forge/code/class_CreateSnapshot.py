import logging
import sys
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from osc_lib import utils
from heatclient.common import format_utils
from heatclient import exc as heat_exc
class CreateSnapshot(command.ShowOne):
    """Create stack snapshot."""
    log = logging.getLogger(__name__ + '.CreateSnapshot')

    def get_parser(self, prog_name):
        parser = super(CreateSnapshot, self).get_parser(prog_name)
        parser.add_argument('stack', metavar='<stack>', help=_('Name or ID of stack'))
        parser.add_argument('--name', metavar='<name>', help=_('Name of snapshot'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        try:
            data = heat_client.stacks.snapshot(parsed_args.stack, parsed_args.name)
        except heat_exc.HTTPNotFound:
            raise exc.CommandError(_('Stack not found: %s') % parsed_args.stack)
        columns = ['ID', 'name', 'status', 'status_reason', 'data', 'creation_time']
        return (columns, utils.get_dict_properties(data, columns))