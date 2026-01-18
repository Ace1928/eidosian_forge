import json
import os
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.queues.v1 import cli
class CreateSubscription(command.ShowOne):
    """Create a subscription for queue"""
    _description = _('Create a subscription for queue')
    log = logging.getLogger(__name__ + '.CreateSubscription')

    def get_parser(self, prog_name):
        parser = super(CreateSubscription, self).get_parser(prog_name)
        parser.add_argument('queue_name', metavar='<queue_name>', help='Name of the queue to subscribe to')
        parser.add_argument('subscriber', metavar='<subscriber>', help='Subscriber which will be notified')
        parser.add_argument('ttl', metavar='<ttl>', type=int, help='Time to live of the subscription in seconds')
        parser.add_argument('--options', type=json.loads, default={}, metavar='<options>', help='Metadata of the subscription in JSON format')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        kwargs = {'options': parsed_args.options}
        if parsed_args.subscriber:
            kwargs['subscriber'] = parsed_args.subscriber
        if parsed_args.subscriber:
            kwargs['ttl'] = parsed_args.ttl
        data = client.subscription(parsed_args.queue_name, **kwargs)
        if not data:
            raise RuntimeError('Failed to create subscription for (%s).' % parsed_args.subscriber)
        columns = ('ID', 'Subscriber', 'TTL', 'Options')
        return (columns, utils.get_item_properties(data, columns))