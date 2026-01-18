import json
import os
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.queues.v1 import cli
class CreateClaim(cli.CreateClaim):
    """Create claim and return a list of claimed messages"""

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        kwargs = {}
        if parsed_args.ttl is not None:
            kwargs['ttl'] = parsed_args.ttl
        if parsed_args.grace is not None:
            kwargs['grace'] = parsed_args.grace
        if parsed_args.limit is not None:
            kwargs['limit'] = parsed_args.limit
        queue = client.queue(parsed_args.queue_name, auto_create=False)
        keys = ('claim_id', 'id', 'ttl', 'age', 'body', 'checksum')
        columns = ('Claim_ID', 'Message_ID', 'TTL', 'Age', 'Messages', 'Checksum')
        data = queue.claim(**kwargs)
        return (columns, (utils.get_item_properties(s, keys) for s in data))