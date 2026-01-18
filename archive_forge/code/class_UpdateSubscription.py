import json
import os
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.queues.v1 import cli
class UpdateSubscription(command.ShowOne):
    """Update a subscription"""
    _description = _('Update a subscription')
    log = logging.getLogger(__name__ + '.UpdateSubscription')

    def get_parser(self, prog_name):
        parser = super(UpdateSubscription, self).get_parser(prog_name)
        parser.add_argument('queue_name', metavar='<queue_name>', help='Name of the queue to subscribe to')
        parser.add_argument('subscription_id', metavar='<subscription_id>', help='ID of the subscription')
        parser.add_argument('--subscriber', metavar='<subscriber>', help='Subscriber which will be notified')
        parser.add_argument('--ttl', metavar='<ttl>', type=int, help='Time to live of the subscription in seconds')
        parser.add_argument('--options', type=json.loads, default={}, metavar='<options>', help='Metadata of the subscription in JSON format')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        data = {'subscriber': parsed_args.subscriber, 'ttl': parsed_args.ttl, 'options': parsed_args.options}
        kwargs = {'id': parsed_args.subscription_id}
        subscription = client.subscription(parsed_args.queue_name, auto_create=False, **kwargs)
        subscription.update(data)
        columns = ('ID', 'Subscriber', 'TTL', 'Options')
        return (columns, utils.get_item_properties(data, columns))