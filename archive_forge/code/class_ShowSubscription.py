import json
import os
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.queues.v1 import cli
class ShowSubscription(command.ShowOne):
    """Display subscription details"""
    _description = _('Display subscription details')
    log = logging.getLogger(__name__ + '.ShowSubscription')

    def get_parser(self, prog_name):
        parser = super(ShowSubscription, self).get_parser(prog_name)
        parser.add_argument('queue_name', metavar='<queue_name>', help='Name of the queue to subscribe to')
        parser.add_argument('subscription_id', metavar='<subscription_id>', help='ID of the subscription')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        kwargs = {'id': parsed_args.subscription_id}
        pool_data = client.subscription(parsed_args.queue_name, **kwargs)
        columns = ('ID', 'Subscriber', 'TTL', 'Age', 'Confirmed', 'Options')
        return (columns, utils.get_dict_properties(pool_data.__dict__, columns))