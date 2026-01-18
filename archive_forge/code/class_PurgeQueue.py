import json
import os
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.queues.v1 import cli
class PurgeQueue(command.Command):
    """Purge a queue"""
    _description = _('Purge a queue')
    log = logging.getLogger(__name__ + '.PurgeQueue')

    def get_parser(self, prog_name):
        parser = super(PurgeQueue, self).get_parser(prog_name)
        parser.add_argument('queue_name', metavar='<queue_name>', help='Name of the queue')
        parser.add_argument('--resource_types', metavar='<resource_types>', action='append', choices=['messages', 'subscriptions'], help='Resource types want to be purged.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        queue_name = parsed_args.queue_name
        client.queue(queue_name).purge(resource_types=parsed_args.resource_types)