import json
import os
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.queues.v1 import cli
class DeleteSubscription(command.Command):
    """Delete a subscription"""
    _description = _('Delete a subscription')
    log = logging.getLogger(__name__ + '.DeleteSubscription')

    def get_parser(self, prog_name):
        parser = super(DeleteSubscription, self).get_parser(prog_name)
        parser.add_argument('queue_name', metavar='<queue_name>', help='Name of the queue for the subscription')
        parser.add_argument('subscription_id', metavar='<subscription_id>', help='ID of the subscription')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        client.subscription(parsed_args.queue_name, id=parsed_args.subscription_id, auto_create=False).delete()