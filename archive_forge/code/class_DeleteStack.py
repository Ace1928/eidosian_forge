import logging
import sys
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib import utils
from oslo_serialization import jsonutils
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import event_utils
from heatclient.common import format_utils
from heatclient.common import hook_utils
from heatclient.common import http
from heatclient.common import template_utils
from heatclient.common import utils as heat_utils
from heatclient import exc as heat_exc
class DeleteStack(command.Command):
    """Delete stack(s)."""
    log = logging.getLogger(__name__ + '.DeleteStack')

    def get_parser(self, prog_name):
        parser = super(DeleteStack, self).get_parser(prog_name)
        parser.add_argument('stack', metavar='<stack>', nargs='+', help=_('Stack(s) to delete (name or ID)'))
        parser.add_argument('-y', '--yes', action='store_true', help=_('Skip yes/no prompt (assume yes)'))
        parser.add_argument('--wait', action='store_true', help=_('Wait for stack delete to complete'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        try:
            if not parsed_args.yes and sys.stdin.isatty():
                prompt_response = input(_('Are you sure you want to delete this stack(s) [y/N]? ')).lower()
                if not prompt_response.startswith('y'):
                    self.log.info('User did not confirm stack delete so taking no action.')
                    return
        except KeyboardInterrupt:
            self.log.info('User did not confirm stack delete (ctrl-c) so taking no action.')
            return
        except EOFError:
            self.log.info('User did not confirm stack delete (ctrl-d) so taking no action.')
            return
        failure_count = 0
        stacks_waiting = []
        for sid in parsed_args.stack:
            marker = None
            if parsed_args.wait:
                try:
                    events = event_utils.get_events(heat_client, stack_id=sid, event_args={'sort_dir': 'desc'}, limit=1)
                    if events:
                        marker = events[0].id
                except heat_exc.CommandError as ex:
                    failure_count += 1
                    print(ex)
                    continue
            try:
                heat_client.stacks.delete(sid)
                stacks_waiting.append((sid, marker))
            except heat_exc.HTTPNotFound:
                failure_count += 1
                print(_('Stack not found: %s') % sid)
            except heat_exc.Forbidden:
                failure_count += 1
                print(_('Forbidden: %s') % sid)
        if parsed_args.wait:
            for sid, marker in stacks_waiting:
                try:
                    stack_status, msg = event_utils.poll_for_events(heat_client, sid, action='DELETE', marker=marker)
                except heat_exc.CommandError:
                    continue
                if stack_status == 'DELETE_FAILED':
                    failure_count += 1
                    print(msg)
        if failure_count:
            msg = _('Unable to delete %(count)d of the %(total)d stacks.') % {'count': failure_count, 'total': len(parsed_args.stack)}
            raise exc.CommandError(msg)